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
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h2("Wood Log Defect Segmentation – 3D Viewer"),

        ui.input_file("tiff_file", "Upload .TIFF log", accept=[".tiff", ".tif"]),
        ui.input_select("model_path", "Select Model Checkpoint", list_model_checkpoints()),

        ui.input_checkbox("use_crf", "Apply CRF postprocessing", value=True),
        ui.input_checkbox("use_mrf", "Apply 3D MRF refinement", value=True),

        ui.input_action_button("run_btn", "Run Segmentation", class_="btn-primary"),
        ui.output_ui("progress_ui"),
    ),

    ui.layout_column_wrap(
        ui.card(
            output_widget("plot", height="90vh"),
            full_screen=True,
            fill=True,
            style="min-height: 80vh;"
        ),
    fill=True
    )
)
# endregion

# region Server
def server(input, output, session):
    status = reactive.Value("Ready.")
    progress = reactive.Value(0)
    fig_val = reactive.Value(None)
    q = queue.Queue()   # messages from worker thread

    # Background worker
    def run_pipeline_background(path, model_path, use_crf, use_mrf):
        try:
            pipe = SegmentationPipeline(model_type="cnn", use_crf=use_crf, use_mrf=use_mrf)

            def progress_callback(current, total):
                pct = int(100 * current / total)
                q.put(("progress", pct, f"Segmenting ({current}/{total})"))

            q.put(("status", "Starting segmentation..."))
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

    # Poll queue regularly
    @reactive.effect
    def _poll_queue():
        reactive.invalidate_later(0.3)
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

    # Start pipeline on click
    @reactive.effect
    @reactive.event(input.run_btn)
    def _():
        fileinfo = input.tiff_file()
        model_path = input.model_path()

        if not fileinfo:
            status.set("No TIFF uploaded.")
            return
        if not model_path or not os.path.exists(model_path):
            status.set("Invalid model selected.")
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

    # Progress Bar UI
    @output
    @render.ui
    def progress_ui():
        pct = progress.get()
        msg = status.get()
        return ui.div(
            ui.div(
                ui.div(
                    f"{pct}%",
                    class_="progress-bar text-white",
                    role="progressbar",
                    style=f"width: {pct}%; transition: width 0.25s;",
                ),
                class_="progress",
                style="height: 20px;",
            ),
            ui.p(msg)
        )

    @output
    @render_widget
    def plot():
        return fig_val.get()
# endregion


app = App(app_ui, server)
