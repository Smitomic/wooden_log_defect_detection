import sys, os, tempfile, shutil, threading, queue, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
from src.pipelines.segmentation_pipeline import SegmentationPipeline


def list_model_checkpoints():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    checkpoints = glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
    if not checkpoints:
        return {"": "No checkpoints found"}
    return {ckpt: os.path.relpath(ckpt, base) for ckpt in checkpoints}


# region UI
def app_ui(request):
    return ui.page_fillable(
        ui.tags.style("""
            .bslib-sidebar-layout {
                flex: 1 1 auto !important;
                height: 0 !important;
                min-height: 0 !important;
            }
            
            /* Remove the automatic vertical gap that bslib inserts in page-fill layouts */
            body.bslib-gap-spacing {
                gap: 0 !important;
            }
            
            /* Remove spacing only from main viewer and not sidebar */
            .main.bslib-gap-spacing {
                padding: 0 !important;
                margin: 0 !important;
                gap: 0 !important;
            }

            /* Ensure plot fills height */
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
                    ui.input_checkbox("use_crf", "Apply CRF postprocessing", value=True),
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
                ui.card(
                    ui.div(
                        output_widget("plot"),
                        style="width:100%; height:100%; min-height:600px; overflow:hidden;"
                    ),
                    full_screen=True,
                    fill=True,
                ),
                fill=True,
            ),
            fillable=True,
            class_="flex-fill h-100 overflow-hidden",
            style="min-height:0;"
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
    q = queue.Queue()

    def run_pipeline_background(path, model_path, use_crf, use_mrf):
        try:
            pipe = SegmentationPipeline(model_type="cnn", use_crf=use_crf, use_mrf=use_mrf)

            def progress_callback(current, total):
                pct = int(100 * current / total)
                q.put(("progress", pct, f"Processing slices ({current}/{total})"))

            q.put(("status", "Segmenting volume..."))
            refined_volume, _ = pipe.run(
                tiff_path=path,
                model_path=model_path,
                visualize=False,
                progress_callback=progress_callback,
            )

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

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff")
        shutil.copyfile(fileinfo[0]["datapath"], tmp.name)

        progress.set(0)
        status.set("Preparing…")
        fig_val.set(None)

        threading.Thread(
            target=run_pipeline_background,
            args=(tmp.name, model_path, input.use_crf(), input.use_mrf()),
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

    @output
    @render_widget
    def plot():
        return fig_val.get()
# endregion

app = App(app_ui, server)
