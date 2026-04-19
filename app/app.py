import os
import shutil
import sys
import tempfile
import threading
import queue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget

from src.pipelines.segmentation_pipeline import MODEL_REGISTRY, SegmentationPipeline
from src.visualization.volume_metrics import CLASS_LABELS, _OLD_CLASS_LABELS



# Model choices for the UI dropdown

MODEL_CHOICES = {
    "New models (5-class)": {
        "DilatedCNN+MRF log-split":    "DilatedCNN+MRF log-split",
        "DilatedCNN+MRF random-split": "DilatedCNN+MRF random-split",
        "UNet++ log-split":            "UNet++ log-split",
        "UNet++ random-split":         "UNet++ random-split",
    },
    "Old models (7-class)": {
        "Old model (no MRF)":  "Old model (no MRF)",
        "Old model (with MRF)": "Old model (with MRF)",
    },
}

MODEL_DESCRIPTIONS = {
    "DilatedCNN+MRF log-split":    "Best crack recall + VS. Log-stratified split.",
    "DilatedCNN+MRF random-split": "Highest precision (fewest false alarms).",
    "UNet++ log-split":            "Best overall (IoU, Recall, VS). Recommended.",
    "UNet++ random-split":         "Good precision/recall balance.",
    "Old model (no MRF)":          "7-class model (Obvod/Hniloba/Dutina/Hrca/Trhlina). No post-processing.",
    "Old model (with MRF)":        "7-class model with 2D MRF refinement.",
}


# region UI
def app_ui(request):
    return ui.page_fillable(
        # JS: clear old Plotly figure before a new run
        ui.tags.script("""
            Shiny.addCustomMessageHandler("clear_plot", function(msg) {
                let container = document.getElementById("plot");
                if (!container) return;
                let old = container.querySelector(".js-plotly-plot");
                if (old) { try { Plotly.purge(old); } catch(e) {} old.remove(); }
            });
        """),

        # CSS
        ui.tags.style("""
            body { background-color: #F6F4EF !important; }

            /* These two rules close the gap - bslib adds bslib-gap-spacing
               to body and to the main content wrapper */
            body.bslib-gap-spacing {
                gap: 0 !important;
            }
            .main.bslib-gap-spacing {
                padding: 0 !important;
                margin: 0 !important;
                gap: 0 !important;
            }

            .bslib-sidebar-layout .sidebar {
                background-color: #F6F4EF !important;
            }
            .bslib-sidebar-layout {
                flex: 1 1 auto !important;
                height: 0 !important;
                min-height: 0 !important;
            }

            #plot, .js-plotly-plot, .plot-container, .svg-container {
                width: 100% !important;
                height: 100% !important;
            }

            .card-body {
                padding: 0 !important;
                margin: 0 !important;
                display: flex !important;
                flex-direction: column !important;
            }

            .model-hint {
                font-size: 0.78rem;
                color: #5A7B6A;
                margin-top: -6px;
                padding: 0 2px 6px 2px;
                line-height: 1.4;
            }

            .metrics-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
            .metrics-table th { background: #3B6E51; color: white; padding: 6px 10px; text-align: left; }
            .metrics-table td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
            .metrics-table tr:last-child td { border-bottom: none; }
            .anom-ok   { color: #3B6E51; }
            .anom-warn { color: #C0392B; font-weight: 600; }
            .shiny-file-input-progress .progress-bar {
                background-color: #3B6E51 !important;
            }

            /* Stretch sidebar panel to full height so border-right reaches bottom */
            .bslib-sidebar-layout > .sidebar {
                min-height: 100% !important;
            }
            .shiny-input-container,
            .shiny-file-input-progress {
                overflow: visible !important;
            }
        """),

        # Header
        ui.div(
            ui.h3("Wood Log Defect Segmentation – 3D Viewer",
                  style="margin:0; font-weight:600;"),
            style=(
                "background-color:#3B6E51; color:white; padding:14px 22px;"
                "border-bottom:4px solid #A67C52; flex:0 0 auto;"
            ),
        ),

        # Main layout
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    ui.h4("Controls", style="color:#3B6E51; font-weight:600;"),

                    ui.input_file(
                        "tiff_file", "Upload .TIFF log",
                        accept=[".tiff", ".tif"],
                    ),
                    ui.input_select(
                        "model_name", "Segmentation model", MODEL_CHOICES,
                        selected="UNet++ log-split",
                    ),
                    ui.output_ui("model_hint"),

                    ui.input_checkbox(
                        "apply_3d_mrf", "Apply 3D MRF refinement",
                        value=True,
                    ),
                    ui.div(
                        "3D MRF smooths label noise across slices. "
                        "Crack class is always frozen to protect hairline structures.",
                        class_="model-hint",
                    ),

                    ui.input_checkbox(
                        "apply_z_close", "Apply z-axis crack closing",
                        value=False,
                    ),
                    ui.div(
                        "Bridges isolated crack fragments across consecutive slices "
                        "without expanding crack width laterally. "
                        "Improves 3D connectivity; IoU stays flat.",
                        class_="model-hint",
                    ),
                    ui.output_ui("z_extent_ui"),

                    style=(
                        "display:flex; flex-direction:column; gap:10px; "
                        "flex:1 1 auto; overflow-x:hidden; overflow-y:auto;"
                    ),
                ),

                # Run button pinned to bottom of sidebar
                ui.div(
                    ui.input_action_button(
                        "run_btn", "Run Segmentation",
                        class_="btn btn-primary",
                        style=(
                            "background-color:#3B6E51; border-color:#3B6E51;"
                            "width:100%; font-weight:600;"
                        ),
                    ),
                    ui.output_ui("progress_ui"),
                    style="margin-top:auto; padding-top:12px; display:flex; flex-direction:column; gap:10px; flex:0 0 auto;",
                ),

                open="always",
                width="280px",
                style=(
                    "display:flex; flex-direction:column; height:100%;"
                    "background-color:#F6F4EF; border-right:2px solid #A67C52; padding:16px;"
                ),
            ),

            # Viewer + metrics - fillable div so bslib distributes height correctly
            ui.div(
                ui.card(
                    output_widget("plot"),
                    full_screen=True,
                    fill=True,
                    style="overflow:hidden;",
                ),
                ui.card(
                    ui.output_ui("metrics_ui"),
                    style="flex:0 0 auto; max-height:220px; overflow:auto; padding:16px;",
                ),
                class_="html-fill-container html-fill-item",
                style=(
                    "display:flex; flex-direction:column; gap:6px;"
                    "height:100%; padding:8px; box-sizing:border-box;"
                ),
            ),
            fillable=True,
            class_="flex-fill h-100",
            style="min-height:0;",
        ),

        class_="d-flex flex-column vh-100 p-0 m-0",
        fillable=True,
        style="min-height:0;",
    )
# endregion

# region Server
def server(input, output, session):
    status   = reactive.Value("Ready.")
    progress = reactive.Value(0)
    fig_val  = reactive.Value(None)
    metrics_val = reactive.Value(None)   # dict: cls -> {metrics, anomalies}
    q        = queue.Queue()

    # Model hint text (updates when selector changes)
    @output
    @render.ui
    def model_hint():
        name = input.model_name()
        hint = MODEL_DESCRIPTIONS.get(name, "")
        return ui.div(hint, class_="model-hint")

    # Background pipeline thread
    def _run_pipeline(
        tiff_path:     str,
        model_name:    str,
        apply_3d_mrf:  bool,
        apply_z_close: bool,
        z_extent:      int,
    ) -> None:
        try:
            pipe = SegmentationPipeline(
                model_name    = model_name,
                apply_3d_mrf  = apply_3d_mrf,
                apply_z_close = apply_z_close,
                z_extent      = z_extent,
            )

            def _progress_cb(current: int, total: int) -> None:
                pct = int(100 * current / total)
                q.put(("progress", pct, f"Processing slices ({current}/{total})"))

            q.put(("status", f"Running {model_name}…"))
            refined_volume, _, metrics, anomalies = pipe.run(
                tiff_path         = tiff_path,
                visualize         = False,
                progress_callback = _progress_cb,
                return_metrics    = True,
            )

            q.put(("metrics", {"metrics": metrics, "anomalies": anomalies}))
            q.put(("status", "Rendering 3D mesh…"))

            from src.visualization.mesh_viewer import show_volume
            fig = show_volume(refined_volume, title=f"3D - {model_name}")

            q.put(("done", fig))

        except Exception as exc:
            q.put(("error", str(exc)))

    # Poll queue every 250 ms
    @reactive.effect
    def _poll():
        reactive.invalidate_later(0.25)
        while not q.empty():
            event = q.get()
            tag   = event[0]

            if tag == "progress":
                _, pct, msg = event
                progress.set(pct)
                status.set(msg)
            elif tag == "status":
                status.set(event[1])
            elif tag == "metrics":
                metrics_val.set(event[1])
            elif tag == "done":
                fig_val.set(event[1])
                progress.set(100)
                status.set("Completed.")
            elif tag == "error":
                status.set("Error: " + event[1])

    # Run button handler
    @reactive.effect
    @reactive.event(input.run_btn)
    def _on_run():
        fileinfo   = input.tiff_file()
        model_name = input.model_name()

        if not fileinfo:
            status.set("No TIFF uploaded.")
            return

        cfg        = MODEL_REGISTRY.get(model_name, {})
        checkpoint = cfg.get("path", "")
        if not os.path.exists(checkpoint):
            status.set(
                f"Checkpoint not found: {checkpoint}. "
                f"Train the '{model_name}' model first."
            )
            return

        # Clear previous results
        fig_val.set(None)
        metrics_val.set(None)
        session.send_custom_message("clear_plot", {})

        # Copy TIFF to temp file (Shiny deletes the upload after the request)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff")
        shutil.copyfile(fileinfo[0]["datapath"], tmp.name)

        progress.set(0)
        status.set("Preparing…")

        threading.Thread(
            target=_run_pipeline,
            args=(
                tmp.name,
                model_name,
                input.apply_3d_mrf(),
                input.apply_z_close(),
                int(input.z_extent()) if input.apply_z_close() else 3,
            ),
            daemon=True,
        ).start()

    # Z-extent selector (only shown when z-closing is enabled)
    @output
    @render.ui
    def z_extent_ui():
        if not input.apply_z_close():
            return ui.div()
        return ui.input_radio_buttons(
            "z_extent", "Closing extent (slices bridged)",
            choices={"3": "z=3  (up to 2 missing)", "5": "z=5  (up to 4 missing)", "7": "z=7  (up to 6 missing)"},
            selected="3",
            inline=False,
        )

    # Progress bar
    @output
    @render.ui
    def progress_ui():
        pct = progress.get()
        msg = status.get()
        return ui.div(
            ui.div(
                ui.div(
                    f"{pct}%",
                    class_="progress-bar",
                    style=(
                        f"width:{pct}%; background-color:#3B6E51; "
                        "color:white; transition:width 0.25s;"
                    ),
                ),
                class_="progress",
                style="height:24px; border:1px solid #A67C52; background:white;",
            ),
            ui.div(msg, style="font-size:0.82rem; margin-top:4px; color:#3B6E51;"),
        )

    # 3D plot
    @output
    @render_widget
    def plot():
        return fig_val.get()

    # Metrics table
    @output
    @render.ui
    def metrics_ui():
        data = metrics_val.get()
        if data is None:
            return ui.div("Run segmentation to see quality metrics.",
                          style="color:#888; font-size:0.85rem;")

        metrics   = data["metrics"]
        anomalies = data["anomalies"]

        # Pick label set based on the currently selected model
        cfg          = MODEL_REGISTRY.get(input.model_name(), {})
        labels_map   = _OLD_CLASS_LABELS if cfg.get("class_scheme") == "old" else CLASS_LABELS

        rows_html = ""
        for cls, name in labels_map.items():
            m = metrics.get(cls)
            if m is None:
                continue
            vol    = m.get("volume_cm3", 0)
            comps  = m.get("components", 0)
            cont   = m.get("continuity")
            comp   = m.get("compactness")

            cont_str = f"{cont:.3f}" if isinstance(cont, float) else "N/A"
            comp_str = f"{comp:.3f}" if isinstance(comp, float) else "N/A"

            rows_html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{vol} cm³</td>
                <td>{comps}</td>
                <td>{cont_str}</td>
                <td>{comp_str}</td>
            </tr>"""

        table_html = f"""
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Volume</th>
                    <th>Components</th>
                    <th>Continuity</th>
                    <th>Compactness</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>"""

        return ui.HTML(table_html)
# endregion

app = App(app_ui, server)