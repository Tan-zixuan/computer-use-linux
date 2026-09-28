//! niri window backend.
//!
//! [niri](https://github.com/YaLTeR/niri) is a scrollable-tiling Wayland
//! compositor that exposes a JSON IPC interface. This backend reads and focuses
//! windows through `niri msg`, mirroring how the Hyprland backend uses
//! `hyprctl`, and falls back to speaking niri's IPC protocol directly over the
//! Unix socket in `NIRI_SOCKET` when the CLI cannot answer. The socket fallback
//! covers hosts where `niri` is not on `PATH` and hosts that did not inherit
//! `NIRI_SOCKET`, which `niri msg` requires.
//!
//! Both transports stay optional: when niri is not the running compositor the
//! CLI fails and the socket is absent, so the backend reports a failure that
//! the registry skips and another compositor backend can answer.

use crate::command_runner;
use crate::terminal::enrich_terminal_windows;
use crate::windowing::registry::BackendProbe;
use crate::windowing::types::{WindowBounds, WindowInfo};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::time::{Duration, SystemTime};

pub const NIRI_BACKEND: &str = "niri";

/// niri answers over a local socket in microseconds. The timeout only exists so
/// a wedged compositor cannot hang `doctor` or `list_windows`.
const NIRI_IPC_TIMEOUT: Duration = Duration::from_secs(2);

/// Raw niri IPC requests. The socket takes JSON; `niri msg` takes argv.
const WINDOWS_REQUEST: &str = "\"Windows\"";
const WINDOWS_CLI: &[&str] = &["msg", "--json", "windows"];
const OUTPUTS_REQUEST: &str = "\"Outputs\"";
const OUTPUTS_CLI: &[&str] = &["msg", "--json", "outputs"];
const WORKSPACES_REQUEST: &str = "\"Workspaces\"";
const WORKSPACES_CLI: &[&str] = &["msg", "--json", "workspaces"];

/// Transport labels used in `doctor` detail strings.
const TRANSPORT_SOCKET: &str = "the niri IPC socket";
const TRANSPORT_CLI: &str = "niri msg";

pub fn probe() -> BackendProbe {
    match list_windows_with_transport() {
        Ok((windows, transport)) => BackendProbe {
            id: NIRI_BACKEND,
            ok: true,
            can_list_windows: true,
            can_focus_apps: true,
            can_focus_windows: true,
            detail: format!("{transport} returned {} window(s)", windows.len()),
        },
        Err(error) => BackendProbe {
            id: NIRI_BACKEND,
            ok: false,
            can_list_windows: false,
            can_focus_apps: false,
            can_focus_windows: false,
            detail: format!("{error:#}"),
        },
    }
}

pub async fn list_windows() -> Result<Vec<WindowInfo>> {
    let (windows, _transport) = tokio::task::spawn_blocking(list_windows_with_transport)
        .await
        .context("niri window listing task panicked")??;
    Ok(windows)
}

/// Focus an exact niri window by id.
///
/// niri replies `{"Ok":"Handled"}` even for an id that does not exist, so a
/// successful return here means "the compositor accepted the action", not "the
/// window is now focused". Callers that need certainty re-query the focused
/// window, which is what the server's focus verification already does.
pub async fn activate_window(window_id: u64) -> Result<()> {
    tokio::task::spawn_blocking(move || activate_window_blocking(window_id))
        .await
        .context("niri focus task panicked")?
}

fn activate_window_blocking(window_id: u64) -> Result<()> {
    let request = format!("{{\"Action\":{{\"FocusWindow\":{{\"id\":{window_id}}}}}}}");
    let cli_args = [
        "msg",
        "action",
        "focus-window",
        "--id",
        &window_id.to_string(),
    ];
    match cli_action(&cli_args) {
        Ok(()) => Ok(()),
        Err(cli_error) => match socket_request(&request)
            .and_then(|reply| ensure_socket_action_succeeded(&request, &reply))
        {
            Ok(()) => Ok(()),
            Err(socket_error) => Err(anyhow!(
                "niri action focus-window --id {window_id} failed: {cli_error:#} (the direct niri IPC fallback also failed: {socket_error:#})"
            )),
        },
    }
}

fn list_windows_with_transport() -> Result<(Vec<WindowInfo>, &'static str)> {
    let reply = request_value(WINDOWS_REQUEST, WINDOWS_CLI, "Windows")?;
    let windows: Vec<NiriWindow> = serde_json::from_value(reply.value)
        .context("failed to parse the niri window list")?;

    // Output geometry is always needed: it supplies the scale factor that turns
    // niri's logical window sizes into the device-pixel space screenshots use.
    // It is fetched even when no window reports a position, because niri leaves
    // `tile_pos_in_workspace_view` null for every window of a scrolling layout
    // while still reporting a non-unit output scale.
    let layout = output_layout().unwrap_or_default();

    let mut windows = windows
        .into_iter()
        .map(|window| window.into_window_info(&layout))
        .collect::<Vec<_>>();
    windows.sort_by_key(|window| window.window_id);
    enrich_terminal_windows(&mut windows);
    Ok((windows, reply.transport))
}

/// Translates niri's logical coordinates into the physical pixel space that
/// screenshots and window crops are expressed in.
///
/// A scaled output is captured as device pixels: on a 2x output a 1536x960
/// logical desktop reports a 3072x1920 `coordinate_width`/`coordinate_height`,
/// so logical window geometry has to be scaled and rebased, just as the
/// Hyprland backend does for wlroots screenshots. Reporting niri's logical
/// numbers unchanged would describe a crop at half the intended size.
#[derive(Debug, Clone, Copy)]
struct NiriCaptureLayout {
    origin_x: i32,
    origin_y: i32,
    scale: f64,
}

impl Default for NiriCaptureLayout {
    /// Pass niri's logical geometry through unchanged when no output geometry
    /// could be read. Size stays useful and positions stay honest.
    fn default() -> Self {
        Self {
            origin_x: 0,
            origin_y: 0,
            scale: 1.0,
        }
    }
}

impl NiriCaptureLayout {
    /// One desktop-wide layout, mirroring the Hyprland backend: the minimum
    /// output origin becomes the capture origin and the largest output scale is
    /// applied to every window.
    fn from_outputs(outputs: &BTreeMap<String, NiriOutputGeometry>) -> Self {
        let geometries = outputs.values().collect::<Vec<_>>();
        let Some(first) = geometries.first() else {
            return Self::default();
        };
        if geometries
            .iter()
            .any(|geometry| !geometry.scale.is_finite() || geometry.scale <= 0.0)
        {
            return Self::default();
        }
        Self {
            origin_x: geometries
                .iter()
                .map(|geometry| geometry.x)
                .min()
                .unwrap_or(first.x),
            origin_y: geometries
                .iter()
                .map(|geometry| geometry.y)
                .min()
                .unwrap_or(first.y),
            scale: geometries
                .iter()
                .map(|geometry| geometry.scale)
                .fold(first.scale, f64::max),
        }
    }

    /// Device-pixel bounds for a logical window rect. A `None` position yields
    /// populated size with `null` x/y rather than dropping bounds entirely.
    fn window_bounds(
        &self,
        position: Option<(f64, f64)>,
        width: f64,
        height: f64,
    ) -> Option<WindowBounds> {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return None;
        }
        Some(WindowBounds {
            x: position.and_then(|(x, _)| self.map_axis(x, self.origin_x)),
            y: position.and_then(|(_, y)| self.map_axis(y, self.origin_y)),
            width: self.map_dimension(width)?,
            height: self.map_dimension(height)?,
        })
    }

    fn map_axis(&self, value: f64, origin: i32) -> Option<i32> {
        round_coordinate((value - f64::from(origin)) * self.scale)
    }

    fn map_dimension(&self, value: f64) -> Option<u32> {
        positive_dimension(value * self.scale)
    }
}

/// niri window geometry, plus the output it belongs to.
#[derive(Debug, Default)]
struct NiriOutputLayout {
    /// Output name -> logical geometry of that output.
    geometries: BTreeMap<String, NiriOutputGeometry>,
    /// Workspace id -> name of the output currently holding it.
    workspace_outputs: BTreeMap<u64, String>,
    capture: NiriCaptureLayout,
}

impl NiriOutputLayout {
    /// Global logical position of a window, when niri reports one.
    ///
    /// `tile_pos_in_workspace_view` is relative to the workspace view of the
    /// output that holds the workspace, so the output's logical origin is added
    /// back in to get a desktop-wide coordinate.
    fn position(&self, window: &NiriWindow) -> Option<(f64, f64)> {
        let local = window.workspace_view_position()?;
        let output = self.workspace_outputs.get(&window.workspace_id?)?;
        let geometry = self.geometries.get(output)?;
        Some((
            f64::from(geometry.x) + local[0],
            f64::from(geometry.y) + local[1],
        ))
    }
}

fn output_layout() -> Option<NiriOutputLayout> {
    let outputs = request_value(OUTPUTS_REQUEST, OUTPUTS_CLI, "Outputs").ok()?;
    let workspaces = request_value(WORKSPACES_REQUEST, WORKSPACES_CLI, "Workspaces").ok()?;
    let geometries = parse_output_geometries(&outputs.value);
    Some(NiriOutputLayout {
        capture: NiriCaptureLayout::from_outputs(&geometries),
        geometries,
        workspace_outputs: parse_workspace_outputs(&workspaces.value),
    })
}

fn parse_output_geometries(outputs: &Value) -> BTreeMap<String, NiriOutputGeometry> {
    let Ok(outputs) = serde_json::from_value::<BTreeMap<String, NiriOutput>>(outputs.clone())
    else {
        return BTreeMap::new();
    };
    outputs
        .into_iter()
        .filter_map(|(name, output)| {
            let logical = output.logical?;
            Some((
                name,
                NiriOutputGeometry {
                    x: logical.x,
                    y: logical.y,
                    scale: logical.scale,
                },
            ))
        })
        .collect()
}

fn parse_workspace_outputs(workspaces: &Value) -> BTreeMap<u64, String> {
    let Ok(workspaces) = serde_json::from_value::<Vec<NiriWorkspace>>(workspaces.clone()) else {
        return BTreeMap::new();
    };
    workspaces
        .into_iter()
        .filter_map(|workspace| {
            let output = workspace.output?;
            Some((workspace.id, output))
        })
        .collect()
}

/// One niri request, plus how it was answered for `doctor` detail strings.
struct NiriReply {
    value: Value,
    transport: &'static str,
}

/// Run a read request, preferring `niri msg` and falling back to the IPC socket.
///
/// `niri msg` comes first so the common path matches the Hyprland backend's
/// `hyprctl` shell-out and the reference niri adapter. The direct IPC fallback
/// is not decorative: `niri msg` refuses to run at all when `NIRI_SOCKET` is
/// unset ("are you running this within niri?"), and it cannot find the
/// pid-suffixed `niri.$WAYLAND_DISPLAY.<pid>.sock` that current niri creates,
/// while the socket can still be derived from `XDG_RUNTIME_DIR`. It also covers
/// hosts where the `niri` binary is simply not on `PATH`.
fn request_value(request: &str, cli_args: &[&str], key: &str) -> Result<NiriReply> {
    match cli_value(cli_args, key) {
        Ok(value) => Ok(NiriReply {
            value,
            transport: TRANSPORT_CLI,
        }),
        Err(cli_error) => match socket_value(request, key) {
            Ok(value) => Ok(NiriReply {
                value,
                transport: TRANSPORT_SOCKET,
            }),
            Err(socket_error) => Err(anyhow!(
                "niri {} failed: {cli_error:#} (the direct niri IPC fallback also failed: {socket_error:#})",
                cli_args.join(" ")
            )),
        },
    }
}

fn cli_value(cli_args: &[&str], key: &str) -> Result<Value> {
    let output = niri_cli(cli_args)?;
    if !output.status.success() {
        bail!("{}", command_detail(&output));
    }
    parse_reply(&String::from_utf8_lossy(&output.stdout), key)
}

fn cli_action(cli_args: &[&str]) -> Result<()> {
    let output = niri_cli(cli_args)?;
    if !output.status.success() {
        bail!("{}", command_detail(&output));
    }
    Ok(())
}

fn socket_value(request: &str, key: &str) -> Result<Value> {
    parse_reply(&socket_request(request)?, key)
}

fn niri_cli(cli_args: &[&str]) -> Result<std::process::Output> {
    let mut command = StdCommand::new("niri");
    command.args(cli_args);
    command_runner::output_blocking(&mut command, "run niri msg")
}

fn command_detail(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let detail = if stderr.is_empty() {
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        stderr
    };
    if detail.is_empty() {
        format!("exit status {}", output.status)
    } else {
        detail
    }
}

/// Send one JSON request over the niri IPC socket and return the raw reply.
fn socket_request(request: &str) -> Result<String> {
    let path = niri_socket_path().context(
        "no niri IPC socket: NIRI_SOCKET is unset or not a socket and no niri.*.sock was found in XDG_RUNTIME_DIR",
    )?;
    let mut stream = UnixStream::connect(&path)
        .with_context(|| format!("failed to connect to the niri IPC socket {}", path.display()))?;
    let _ = stream.set_read_timeout(Some(NIRI_IPC_TIMEOUT));
    let _ = stream.set_write_timeout(Some(NIRI_IPC_TIMEOUT));

    // niri answers only once it can tell the request is complete, so terminate
    // with a newline and half-close the write side.
    stream
        .write_all(request.as_bytes())
        .and_then(|()| stream.write_all(b"\n"))
        .and_then(|()| stream.flush())
        .with_context(|| format!("failed to send the niri IPC request {request}"))?;
    let _ = stream.shutdown(std::net::Shutdown::Write);

    let mut reply = String::new();
    stream
        .read_to_string(&mut reply)
        .with_context(|| format!("failed to read the niri IPC reply to {request}"))?;
    Ok(reply)
}

/// Unwrap a niri reply.
///
/// The socket wraps every answer as `{"Ok": <payload>}` or `{"Err": "..."}`
/// with the payload nested under a request-named key (`{"Windows": [...]}`),
/// while `niri msg --json` prints only the bare payload. Both shapes normalise
/// to the payload here.
fn parse_reply(raw: &str, key: &str) -> Result<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("the niri IPC reply was empty");
    }
    let value: Value = serde_json::from_str(trimmed)
        .with_context(|| format!("failed to parse the niri reply {trimmed}"))?;
    if let Some(error) = value.get("Err").and_then(Value::as_str) {
        bail!("niri IPC returned an error: {error}");
    }
    let value = match value {
        Value::Object(mut map) => map.remove("Ok").unwrap_or(Value::Object(map)),
        other => other,
    };
    match value {
        Value::Object(mut map) => Ok(match map.remove(key) {
            Some(payload) => payload,
            None => Value::Object(map),
        }),
        other => Ok(other),
    }
}

/// Validate the reply to an `Action` request.
fn ensure_socket_action_succeeded(request: &str, reply: &str) -> Result<()> {
    let trimmed = reply.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        return Ok(());
    };
    match value.get("Err").and_then(Value::as_str) {
        Some(error) => bail!("niri IPC returned an error for {request}: {error}"),
        None => Ok(()),
    }
}

/// Path of the running niri instance's IPC socket.
fn niri_socket_path() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("NIRI_SOCKET") {
        let path = PathBuf::from(value);
        if is_socket(&path) {
            return Some(path);
        }
    }
    infer_niri_socket_path()
}

/// Recover the socket path when `NIRI_SOCKET` is missing.
///
/// niri itself falls back to `$XDG_RUNTIME_DIR/niri.$WAYLAND_DISPLAY.sock`, but
/// current versions append the compositor pid
/// (`niri.wayland-1.5081.sock`), so the exact name is tried first and then a
/// pid-suffixed scan, mirroring how the Hyprland backend infers its instance
/// signature.
fn infer_niri_socket_path() -> Option<PathBuf> {
    let runtime = xdg_runtime_dir()?;
    let display = std::env::var("WAYLAND_DISPLAY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    if let Some(display) = display.as_deref() {
        let exact = runtime.join(format!("niri.{display}.sock"));
        if is_socket(&exact) {
            return Some(exact);
        }
    }

    let candidates = fs::read_dir(&runtime)
        .ok()?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let name = path.file_name()?.to_string_lossy().into_owned();
            niri_socket_candidate(&path, &name, display.as_deref())
        })
        .collect::<Vec<_>>();

    select_niri_socket(candidates).map(|candidate| candidate.path)
}

fn is_socket(path: &Path) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.file_type().is_socket())
        .unwrap_or(false)
}

/// Whether a `niri.<stem>.sock` name belongs to the given `WAYLAND_DISPLAY`.
///
/// `wayland-1` matches both the plain `niri.wayland-1.sock` and the
/// pid-suffixed `niri.wayland-1.<pid>.sock` that current niri uses, but never a
/// different display such as `wayland-10`.
fn niri_socket_name_matches_display(stem: &str, display: Option<&str>) -> bool {
    display.is_some_and(|display| {
        stem == display || stem.strip_prefix(display).is_some_and(|rest| rest.starts_with('.'))
    })
}

fn niri_socket_candidate(
    path: &Path,
    name: &str,
    display: Option<&str>,
) -> Option<NiriSocketCandidate> {
    let stem = name.strip_prefix("niri.")?.strip_suffix(".sock")?;
    if !is_socket(path) {
        return None;
    }
    let display_matches = niri_socket_name_matches_display(stem, display);
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    Some(NiriSocketCandidate {
        path: path.to_path_buf(),
        display_matches,
        modified,
    })
}

fn select_niri_socket(candidates: Vec<NiriSocketCandidate>) -> Option<NiriSocketCandidate> {
    candidates
        .into_iter()
        .max_by_key(|candidate| (candidate.display_matches, candidate.modified))
}

fn xdg_runtime_dir() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("XDG_RUNTIME_DIR") {
        return Some(PathBuf::from(value));
    }
    let uid = fs::metadata("/proc/self").ok()?.uid();
    Some(PathBuf::from(format!("/run/user/{uid}")))
}

#[derive(Debug)]
struct NiriSocketCandidate {
    path: PathBuf,
    display_matches: bool,
    modified: SystemTime,
}

#[derive(Debug, Deserialize)]
struct NiriWindow {
    id: u64,
    title: Option<String>,
    app_id: Option<String>,
    pid: Option<i64>,
    workspace_id: Option<u64>,
    #[serde(default)]
    is_focused: bool,
    #[serde(default)]
    is_minimized: bool,
    #[serde(default)]
    layout: Option<NiriWindowLayout>,
}

#[derive(Debug, Deserialize)]
struct NiriWindowLayout {
    window_size: Option<[f64; 2]>,
    tile_size: Option<[f64; 2]>,
    /// Position inside the workspace view, in logical coordinates. niri leaves
    /// this `null` for windows whose workspace view is not being rendered.
    tile_pos_in_workspace_view: Option<[f64; 2]>,
}

#[derive(Debug, Deserialize)]
struct NiriOutput {
    logical: Option<NiriOutputLogical>,
}

#[derive(Debug, Deserialize)]
struct NiriOutputLogical {
    x: i32,
    y: i32,
    /// Output scale factor. niri always reports it, but a missing value must not
    /// drop the output, so it falls back to 1.0.
    #[serde(default = "unit_scale")]
    scale: f64,
}

fn unit_scale() -> f64 {
    1.0
}

/// Logical geometry of one output.
#[derive(Debug, Clone, Copy, PartialEq)]
struct NiriOutputGeometry {
    x: i32,
    y: i32,
    scale: f64,
}

#[derive(Debug, Deserialize)]
struct NiriWorkspace {
    id: u64,
    output: Option<String>,
}

impl NiriWindow {
    fn workspace_view_position(&self) -> Option<[f64; 2]> {
        self.layout.as_ref()?.tile_pos_in_workspace_view
    }

    /// Window size in niri's logical pixels, preferring the window's own size
    /// over the tile it sits in.
    fn logical_size(&self) -> Option<(f64, f64)> {
        let layout = self.layout.as_ref()?;
        let [width, height] = layout.window_size.or(layout.tile_size)?;
        Some((width, height))
    }

    fn into_window_info(self, layout: &NiriOutputLayout) -> WindowInfo {
        let bounds = self.logical_size().and_then(|(width, height)| {
            layout
                .capture
                .window_bounds(layout.position(&self), width, height)
        });
        WindowInfo {
            window_id: self.id,
            title: self.title,
            app_id: self.app_id.clone(),
            wm_class: self.app_id,
            pid: self.pid.and_then(|pid| u32::try_from(pid).ok()),
            bounds,
            workspace: self
                .workspace_id
                .and_then(|workspace| i32::try_from(workspace).ok()),
            focused: self.is_focused,
            // niri's `is_minimized` is the window the rest of the server means
            // by hidden, and minimized windows report no workspace.
            hidden: self.is_minimized,
            client_type: None,
            backend: NIRI_BACKEND.to_string(),
            terminal: None,
        }
    }
}

fn positive_dimension(value: f64) -> Option<u32> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let rounded = value.round();
    if rounded > f64::from(u32::MAX) {
        return None;
    }
    Some(rounded as u32)
}

fn round_coordinate(value: f64) -> Option<i32> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return None;
    }
    Some(rounded as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed capture of `niri msg --json windows` from a live niri 26.04
    /// session: a focused tiled window, a tile without a rendered position, and
    /// a minimized window with no workspace.
    const LIVE_WINDOWS: &str = r#"[
        {"id":41,"title":"dsh web ~","app_id":"com.mitchellh.ghostty","pid":159272,
         "workspace_id":6,"is_focused":false,"is_floating":false,"is_minimized":false,
         "is_urgent":false,"layout":{"pos_in_scrolling_layout":[1,1],
         "tile_size":[744.0,876.0],"window_size":[744,876],
         "tile_pos_in_workspace_view":null,"window_offset_in_tile":[0.0,0.0]}},
        {"id":48,"title":"Zen Browser","app_id":"zen","pid":194746,
         "workspace_id":6,"is_focused":true,"is_floating":false,"is_minimized":false,
         "is_urgent":false,"layout":{"pos_in_scrolling_layout":[2,1],
         "tile_size":[1536.0,908.0],"window_size":[1536,908],
         "tile_pos_in_workspace_view":null,"window_offset_in_tile":[0.0,0.0]}},
        {"id":37,"title":"notes.md","app_id":"dev.zed.Zed","pid":156824,
         "workspace_id":null,"is_focused":false,"is_floating":false,"is_minimized":true,
         "is_urgent":false,"layout":{"pos_in_scrolling_layout":null,
         "tile_size":[1536.0,908.0],"window_size":[1536,908],
         "tile_pos_in_workspace_view":null,"window_offset_in_tile":[0.0,0.0]}}
    ]"#;

    const LIVE_OUTPUTS: &str = r#"{
        "eDP-1":{"name":"eDP-1","logical":{"x":0,"y":0,"width":1536,"height":960,"scale":2.0}}
    }"#;

    const LIVE_WORKSPACES: &str = r#"[
        {"id":6,"idx":1,"name":null,"output":"eDP-1","is_active":true,"is_focused":true},
        {"id":7,"idx":2,"name":null,"output":null,"is_active":false,"is_focused":false}
    ]"#;

    fn parse_windows(json: &str, layout: &NiriOutputLayout) -> Vec<WindowInfo> {
        serde_json::from_str::<Vec<NiriWindow>>(json)
            .expect("windows fixture should parse")
            .into_iter()
            .map(|window| window.into_window_info(layout))
            .collect()
    }

    fn live_layout() -> NiriOutputLayout {
        let geometries = parse_output_geometries(
            &serde_json::from_str(LIVE_OUTPUTS).expect("outputs fixture"),
        );
        NiriOutputLayout {
            capture: NiriCaptureLayout::from_outputs(&geometries),
            geometries,
            workspace_outputs: parse_workspace_outputs(
                &serde_json::from_str(LIVE_WORKSPACES).expect("workspaces fixture"),
            ),
        }
    }

    /// Build a layout from a units-scale set of outputs so position maths is
    /// readable in the tests that are not about scaling.
    fn layout_at_unit_scale(
        geometries: &[(&str, i32, i32)],
        workspaces: &[(u64, &str)],
    ) -> NiriOutputLayout {
        let geometries = geometries
            .iter()
            .map(|(name, x, y)| {
                (
                    (*name).to_string(),
                    NiriOutputGeometry {
                        x: *x,
                        y: *y,
                        scale: 1.0,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        NiriOutputLayout {
            capture: NiriCaptureLayout::from_outputs(&geometries),
            geometries,
            workspace_outputs: workspaces
                .iter()
                .map(|(id, output)| (*id, (*output).to_string()))
                .collect(),
        }
    }

    #[test]
    fn reads_id_app_id_pid_focus_and_size_from_a_live_capture() {
        let windows = parse_windows(LIVE_WINDOWS, &live_layout());
        assert_eq!(
            windows
                .iter()
                .map(|window| window.window_id)
                .collect::<Vec<_>>(),
            [41, 48, 37]
        );

        let zen = &windows[1];
        assert_eq!(zen.app_id.as_deref(), Some("zen"));
        assert_eq!(zen.title.as_deref(), Some("Zen Browser"));
        assert_eq!(zen.pid, Some(194746));
        assert!(zen.focused);
        assert_eq!(zen.workspace, Some(6));
        assert_eq!(zen.backend, NIRI_BACKEND);

        // niri reports logical 1536x908; the 2x output is captured as device
        // pixels, so the bounds must describe a 3072x1816 rect.
        let bounds = zen.bounds.as_ref().expect("size is always reported");
        assert_eq!((bounds.width, bounds.height), (3072, 1816));
    }

    #[test]
    fn scales_logical_geometry_into_the_screenshot_coordinate_space() {
        // Regression guard: `computer-use-linux screenshot` reports
        // coordinate_width/height 3072x1920 for this 1536x960 logical output.
        // Unscaled logical bounds would crop at half the intended size.
        let layout = live_layout();
        assert_eq!(layout.capture.scale, 2.0);

        let windows = parse_windows(
            r#"[{"id":1,"app_id":"a","workspace_id":6,
                 "layout":{"window_size":[1536,908],"tile_pos_in_workspace_view":[10.0,20.0]}}]"#,
            &layout,
        );
        let bounds = windows[0].bounds.as_ref().unwrap();
        assert_eq!((bounds.width, bounds.height), (3072, 1816));
        assert_eq!((bounds.x, bounds.y), (Some(20), Some(40)));
    }

    #[test]
    fn passes_logical_geometry_through_when_no_output_is_known() {
        // Without output geometry there is no scale to apply, so bounds stay in
        // niri's logical numbers instead of being fabricated.
        let windows = parse_windows(LIVE_WINDOWS, &NiriOutputLayout::default());
        let bounds = windows[1].bounds.as_ref().unwrap();
        assert_eq!((bounds.width, bounds.height), (1536, 908));
    }

    #[test]
    fn keeps_position_null_while_niri_does_not_render_the_workspace_view() {
        // Every window in the live capture has tile_pos_in_workspace_view null,
        // even the focused one; bounds must degrade to size-only, not to zero.
        let windows = parse_windows(LIVE_WINDOWS, &live_layout());
        for window in &windows {
            let bounds = window.bounds.as_ref().expect("size is always reported");
            assert_eq!(bounds.x, None, "window {} reported x", window.window_id);
            assert_eq!(bounds.y, None, "window {} reported y", window.window_id);
        }
    }

    #[test]
    fn offsets_a_rendered_position_by_its_output_origin() {
        let layout = layout_at_unit_scale(
            &[("eDP-1", 0, 0), ("HDMI-A-1", 1920, 180)],
            &[(6, "eDP-1"), (9, "HDMI-A-1")],
        );
        let json = r#"[
            {"id":1,"app_id":"a","workspace_id":6,
             "layout":{"window_size":[800,600],"tile_pos_in_workspace_view":[100.4,49.6]}},
            {"id":2,"app_id":"b","workspace_id":9,
             "layout":{"window_size":[800,600],"tile_pos_in_workspace_view":[10.0,20.0]}},
            {"id":3,"app_id":"c","workspace_id":null,
             "layout":{"window_size":[800,600],"tile_pos_in_workspace_view":[10.0,20.0]}}
        ]"#;
        let windows = parse_windows(json, &layout);

        let first = windows[0].bounds.as_ref().unwrap();
        assert_eq!((first.x, first.y), (Some(100), Some(50)));
        let second = windows[1].bounds.as_ref().unwrap();
        assert_eq!((second.x, second.y), (Some(1930), Some(200)));
        // No workspace means the output is unknown, so a reported view position
        // cannot be promoted to a desktop coordinate.
        let third = windows[2].bounds.as_ref().unwrap();
        assert_eq!((third.x, third.y), (None, None));
    }

    #[test]
    fn rebases_the_y_axis_by_its_own_origin() {
        // Guards against subtracting origin_x from the y axis: both axes are
        // mapped by the same helper, so the origin has to be passed per axis.
        let layout = layout_at_unit_scale(
            &[("eDP-1", 0, 100), ("HDMI-A-1", 0, 300)],
            &[(6, "eDP-1"), (9, "HDMI-A-1")],
        );
        assert_eq!(layout.capture.origin_y, 100);

        let json = r#"[
            {"id":1,"app_id":"a","workspace_id":6,
             "layout":{"window_size":[800,600],"tile_pos_in_workspace_view":[10.0,20.0]}},
            {"id":2,"app_id":"b","workspace_id":9,
             "layout":{"window_size":[800,600],"tile_pos_in_workspace_view":[10.0,20.0]}}
        ]"#;
        let windows = parse_windows(json, &layout);

        let first = windows[0].bounds.as_ref().unwrap();
        assert_eq!((first.x, first.y), (Some(10), Some(20)));
        // output y 300 + local 20 - capture origin 100 = 220.
        let second = windows[1].bounds.as_ref().unwrap();
        assert_eq!((second.x, second.y), (Some(10), Some(220)));
    }

    #[test]
    fn falls_back_to_the_tile_size_when_window_size_is_missing() {
        let json = r#"[
            {"id":1,"app_id":"a","layout":{"tile_size":[1200.6,700.2]}},
            {"id":2,"app_id":"b","layout":{"tile_size":[0.0,700.0]}},
            {"id":3,"app_id":"c"},
            {"id":4,"app_id":"d","layout":{"window_size":[800,600],"tile_size":[1.0,1.0]}}
        ]"#;
        let windows = parse_windows(json, &NiriOutputLayout::default());

        let bounds = windows[0].bounds.as_ref().unwrap();
        assert_eq!((bounds.width, bounds.height), (1201, 700));
        // A zero dimension is not a usable rect, so bounds are omitted entirely.
        assert!(windows[1].bounds.is_none());
        assert!(windows[2].bounds.is_none());
        // window_size wins over tile_size.
        assert_eq!(windows[3].bounds.as_ref().unwrap().width, 800);
    }

    #[test]
    fn a_minimized_window_is_hidden_and_loses_its_workspace() {
        let windows = parse_windows(LIVE_WINDOWS, &live_layout());
        let zed = &windows[2];
        assert!(zed.hidden, "is_minimized should map to hidden");
        assert_eq!(zed.workspace, None);
        assert!(!zed.focused);
        assert!(!windows[1].hidden);
    }

    #[test]
    fn parses_both_the_socket_envelope_and_the_bare_cli_payload() {
        let socket_reply = r#"{"Ok":{"Windows":[{"id":41}]}}"#;
        let cli_payload = r#"[{"id":41}]"#;
        let expected = serde_json::json!([{"id":41}]);

        assert_eq!(parse_reply(socket_reply, "Windows").unwrap(), expected);
        assert_eq!(parse_reply(cli_payload, "Windows").unwrap(), expected);
        // niri terminates socket replies with a newline.
        assert_eq!(
            parse_reply(&format!("{socket_reply}\n"), "Windows").unwrap(),
            expected
        );
    }

    #[test]
    fn surfaces_the_error_envelope_that_niri_returns_for_bad_requests() {
        let error = parse_reply(r#"{"Err":"error parsing request"}"#, "Windows")
            .expect_err("an Err envelope must not parse as a window list");
        assert!(format!("{error:#}").contains("error parsing request"));

        assert!(parse_reply("   ", "Windows").is_err());
        assert!(parse_reply("not json", "Windows").is_err());
    }

    #[test]
    fn tolerates_actions_that_niri_answers_with_handled() {
        assert!(ensure_socket_action_succeeded("focus", r#"{"Ok":"Handled"}"#).is_ok());
        assert!(ensure_socket_action_succeeded("focus", "").is_ok());
        assert!(ensure_socket_action_succeeded(
            "focus",
            r#"{"Err":"cannot focus: no such window"}"#
        )
        .is_err());
    }

    #[test]
    fn parses_output_geometries_and_ignores_outputs_without_logical_geometry() {
        let geometries = parse_output_geometries(
            &serde_json::from_str(
                r#"{
                    "eDP-1":{"logical":{"x":0,"y":0,"width":1536,"height":960,"scale":2.0}},
                    "HDMI-A-1":{"logical":{"x":-1920,"y":100,"width":1920,"height":1080,"scale":1.0}},
                    "DP-1":{"name":"DP-1"}
                }"#,
            )
            .unwrap(),
        );
        let edp = geometries.get("eDP-1").expect("eDP-1 should be parsed");
        assert_eq!((edp.x, edp.y, edp.scale), (0, 0, 2.0));
        let hdmi = geometries.get("HDMI-A-1").expect("HDMI-A-1 should be parsed");
        assert_eq!((hdmi.x, hdmi.y, hdmi.scale), (-1920, 100, 1.0));
        assert_eq!(geometries.get("DP-1"), None);
    }

    #[test]
    fn capture_layout_rebases_to_the_minimum_origin_and_maximum_scale() {
        let geometries = parse_output_geometries(
            &serde_json::from_str(
                r#"{
                    "eDP-1":{"logical":{"x":0,"y":0,"scale":2.0}},
                    "HDMI-A-1":{"logical":{"x":-1920,"y":100,"scale":1.0}}
                }"#,
            )
            .unwrap(),
        );
        let capture = NiriCaptureLayout::from_outputs(&geometries);
        assert_eq!((capture.origin_x, capture.origin_y), (-1920, 0));
        assert_eq!(capture.scale, 2.0);
    }

    #[test]
    fn capture_layout_rejects_an_unusable_output_scale() {
        let geometries = parse_output_geometries(
            &serde_json::from_str(
                r#"{"eDP-1":{"logical":{"x":0,"y":0,"scale":0.0}}}"#,
            )
            .unwrap(),
        );
        let capture = NiriCaptureLayout::from_outputs(&geometries);
        assert_eq!(capture.scale, 1.0);
        assert_eq!((capture.origin_x, capture.origin_y), (0, 0));
        assert_eq!(
            NiriCaptureLayout::from_outputs(&BTreeMap::new()).scale,
            1.0
        );
    }

    #[test]
    fn reads_the_display_name_out_of_a_niri_socket_filename() {
        assert!(niri_socket_name_matches_display("wayland-1", Some("wayland-1")));
        // Current niri appends the compositor pid.
        assert!(niri_socket_name_matches_display(
            "wayland-1.5081",
            Some("wayland-1")
        ));
        // A different display must never match, including the wayland-1 /
        // wayland-10 prefix trap.
        assert!(!niri_socket_name_matches_display(
            "wayland-10.99",
            Some("wayland-1")
        ));
        assert!(!niri_socket_name_matches_display("wayland-2", Some("wayland-1")));
        assert!(!niri_socket_name_matches_display("wayland-1", None));
    }

    #[test]
    fn parses_workspace_outputs_and_ignores_detached_workspaces() {
        let outputs = parse_workspace_outputs(
            &serde_json::from_str(LIVE_WORKSPACES).unwrap(),
        );
        assert_eq!(outputs.get(&6).map(String::as_str), Some("eDP-1"));
        assert_eq!(outputs.get(&7), None);
    }

    #[test]
    fn accepts_a_real_niri_socket_and_rejects_other_runtime_entries() {
        let dir = std::env::temp_dir().join(format!("cul-niri-socket-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir should be creatable");

        let socket = dir.join("niri.wayland-1.5081.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&socket)
            .expect("a unix socket should be bindable");
        let plain = dir.join("niri.lock");
        fs::write(&plain, b"").expect("a plain file should be writable");
        let other = dir.join("other.sock");
        let _other_listener = std::os::unix::net::UnixListener::bind(&other)
            .expect("a unix socket should be bindable");

        let candidate = niri_socket_candidate(&socket, "niri.wayland-1.5081.sock", Some("wayland-1"))
            .expect("a matching niri socket should be a candidate");
        assert!(candidate.display_matches);

        let mismatched = niri_socket_candidate(&socket, "niri.wayland-1.5081.sock", Some("wayland-9"))
            .expect("the socket is still a candidate for another display");
        assert!(!mismatched.display_matches);

        // Not a socket, or not a `niri.*.sock` name at all.
        assert!(niri_socket_candidate(&plain, "niri.lock", Some("wayland-1")).is_none());
        assert!(niri_socket_candidate(&other, "other.sock", Some("wayland-1")).is_none());
        assert!(niri_socket_candidate(&dir, "niri.dir.sock", Some("wayland-1")).is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prefers_the_socket_of_the_matching_wayland_display() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let older = SystemTime::UNIX_EPOCH;
        let candidates = vec![
            NiriSocketCandidate {
                path: PathBuf::from("/run/user/1000/niri.wayland-2.5.sock"),
                display_matches: false,
                modified: now,
            },
            NiriSocketCandidate {
                path: PathBuf::from("/run/user/1000/niri.wayland-1.9.sock"),
                display_matches: true,
                modified: older,
            },
        ];
        let selected = select_niri_socket(candidates).expect("a candidate should win");
        assert_eq!(
            selected.path,
            PathBuf::from("/run/user/1000/niri.wayland-1.9.sock")
        );
        assert!(select_niri_socket(Vec::new()).is_none());
    }
}
