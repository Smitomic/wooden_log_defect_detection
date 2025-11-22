import sys, os, tempfile, shutil, threading, queue, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
from src.pipelines.segmentation_pipeline import SegmentationPipeline
from src.visualization.volume_metrics import CLASS_LABELS

def list_model_checkpoints():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    checkpoints = glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
    if not checkpoints:
        return {"": "No checkpoints found"}
    return {ckpt: os.path.relpath(ckpt, base) for ckpt in checkpoints}


# region UI
def app_ui(request):
    return ui.page_fillable(
        ui.tags.script("""
               Shiny.addCustomMessageHandler("clear_plot", function(msg) {
                   let container = document.getElementById("plot");
                   if (!container) return;

                   // Find the internal plot div
                   let old_plot = container.querySelector('.js-plotly-plot');
                   if (old_plot) {
                       try {
                           Plotly.purge(old_plot);   // free GPU memory
                       } catch(e) {}
                       old_plot.remove();            // remove DOM node
                   }
               });
           """),

        ui.tags.style("""
            .bslib-sidebar-layout {
                flex: 1 1 auto !important;
                height: 0 !important;
                min-height: 0 !important;
            }
            
            body.bslib-gap-spacing {
                gap: 0 !important;
            }

            .main.bslib-gap-spacing {
                padding: 0 !important;
                margin: 0 !important;
                gap: 0 !important;
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

            .shiny-file-input-progress {
                width: 100% !important;
                display: block !important;
                margin-top: 4px;
            }

            .shiny-file-input-progress .progress-bar {
                width: 100% !important;
            }

            .shiny-input-container {
                overflow: visible !important;
            }

            .input-group {
                width: 100% !important;
                display: flex !important;
            }

            .shiny-file-input-progress {
                width: 100% !important;
                margin-top: 6px !important;
                overflow: visible !important;
            }

            .shiny-file-input-progress .progress-bar {
                background-color: #3B6E51 !important;
            }
        """),

    # HEADER
        ui.div(
            ui.h3(
                "Wood Log Defect Segmentation – 3D Viewer",
                style="margin:0; font-weight:600;"
            ),
            style=(
                "background-color:#3B6E51; color:white; padding:14px 22px;"
                "border-bottom:4px solid #A67C52; flex:0 0 auto;"
            ),
        ),

        # MAIN (sidebar + viewer)
        ui.layout_sidebar(
            # Sidebar
            ui.sidebar(
                ui.div(
                    ui.h4("Controls", style="color:#3B6E51; font-weight:600;"),
                    ui.input_file("tiff_file", "Upload .TIFF log", accept=[".tiff", ".tif"]),
                    ui.input_select("model_path", "Model Checkpoint", list_model_checkpoints()),
                    ui.input_checkbox("use_mrf", "Apply 3D MRF refinement", value=True),
                    style="display:flex; flex-direction:column; gap:12px; flex:1 1 auto; overflow:auto;"
                ),

                ui.div(
                    ui.input_action_button(
                        "run_btn", "Run Segmentation",
                        class_="btn btn-primary",
                        style="background-color:#3B6E51; border-color:#3B6E51; width:100%; font-weight:600;"
                    ),
                    ui.output_ui("progress_ui"),
                    # gap between button and progress + pin block to bottom
                    style="margin-top:auto; padding-top:12px; display:flex; flex-direction:column; gap:14px;"
                ),

                open="always",
                width="260px",
                style=(
                    "display:flex; flex-direction:column; height:100%;"
                    "background-color:#F6F4EF; border-right:2px solid #A67C52; padding:16px;"
                ),
            ),

            # Viewer
            ui.layout_column_wrap(
                ui.div(
                    ui.card(
                        ui.div(
                            output_widget("plot"),
                            style="width:100%; height:100%; min-height:850px; overflow:hidden;"
                        ),
                        full_screen=True,
                        fill=True,
                    ),
                    ui.card(
                        ui.pre(ui.output_text("metrics_text")),
                        style="padding:20px; max-height:300px; overflow:auto;"
                    ),
                    style="display:flex; flex-direction:column; gap:5px; width:100%;"
                ),
                fill=True,
            ),
            fillable=True,
            class_="flex-fill h-100",
            style="min-height:0; overflow-y:auto;"
        ),
        class_="d-flex flex-column vh-100 p-0 m-0",
        fillable=True,
        style="min-height:0;"
    )
# endregion

# region SERVER
def server(input, output, session):
    status = reactive.Value("Ready.")
    progress = reactive.Value(0)
    fig_val = reactive.Value(None)
    metrics_text = reactive.Value("")
    q = queue.Queue()

    def run_pipeline_background(path, model_path, use_mrf):
        try:
            pipe = SegmentationPipeline(model_type="cnn", use_mrf=use_mrf)

            def progress_callback(current, total):
                pct = int(100 * current / total)
                q.put(("progress", pct, f"Processing slices ({current}/{total})"))

            q.put(("status", "Segmenting volume..."))
            refined_volume, _, metrics, anomalies = pipe.run(
                tiff_path=path,
                model_path=model_path,
                visualize=False,
                progress_callback=progress_callback,
                return_metrics=True,
            )

            # Format metrics into readable text
            summary = []
            for cls, m in metrics.items():
                name = CLASS_LABELS.get(cls, f"Class {cls}")

                vol = m.get("volume_cm3")
                comps = m.get("components")
                cont = m.get("continuity")
                comp = m.get("compactness")

                # Safe conversions
                vol_str = f"{vol}" if vol is not None else "0"
                comps_str = f"{comps}" if comps is not None else "0"
                cont_str = f"{cont:.3f}" if isinstance(cont, (float, int)) else "N/A"
                comp_str = f"{comp:.3f}" if isinstance(comp, (float, int)) else "N/A"

                anoms = anomalies.get(cls, [])
                anom_str = "; ".join(anoms) if anoms else "OK"

                summary.append(
                    f"**{name}** — Volume: {vol_str} cm³ · Components: {comps_str} · "
                    f"Continuity: {cont_str} · Compactness: {comp_str} — {anom_str}"
                )

            q.put(("metrics", "\n".join(summary)))


            q.put(("status", "Rendering 3D mesh..."))
            from src.visualization.mesh_viewer import show_volume
            fig = show_volume(refined_volume)

            q.put(("done", fig))

        except Exception as e:
            q.put(("error", str(e)))

    @reactive.effect
    def _poll():
        reactive.invalidate_later(0.25)
        while not q.empty():
            event = q.get()

            if event[0] == "progress":
                _, pct, msg = event
                progress.set(pct)
                status.set(msg)

            elif event[0] == "status":
                _, msg = event
                status.set(msg)

            elif event[0] == "done":
                _, fig = event
                fig_val.set(fig)
                progress.set(100)
                status.set("Completed.")

            elif event[0] == "metrics":
                _, txt = event
                metrics_text.set(txt)

            elif event[0] == "error":
                _, msg = event
                status.set("Error: " + msg)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _():
        fileinfo = input.tiff_file()
        model_path = input.model_path()

        if not fileinfo:
            status.set("No TIFF uploaded.")
            return
        if not os.path.exists(model_path):
            status.set("Invalid model checkpoint.")
            return

        # Purge old plot
        fig_val.set(None)
        metrics_text.set("")
        session.send_custom_message("clear_plot", {})

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff")
        shutil.copyfile(fileinfo[0]["datapath"], tmp.name)

        progress.set(0)
        status.set("Preparing…")

        threading.Thread(
            target=run_pipeline_background,
            args=(tmp.name, model_path, input.use_mrf()),
            daemon=True,
        ).start()

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
                        f"width:{pct}%; background-color:#3B6E51;"
                        "color:white; transition:width 0.25s;"
                    )
                ),
                class_="progress",
                style="height:24px; border:1px solid #A67C52; background:white;"
            ),
            ui.div(msg, style="font-size:0.85rem; margin-top:4px; color:#3B6E51;")
        )

    # 3D plot output
    @output
    @render_widget
    def plot():
        return fig_val.get()

    # Metrics output
    @output(id="metrics_text")
    @render.text
    def metrics_text_out():
        return metrics_text.get()
# endregion

app = App(app_ui, server)
