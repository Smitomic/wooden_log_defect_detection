import sys, os, tempfile, shutil, threading, glob, asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shiny import App, ui, reactive
from shinywidgets import output_widget, render_widget

from src.pipelines.segmentation_pipeline import SegmentationPipeline


def list_model_checkpoints():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    models = glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
    if not models:
        return {"": "No model checkpoints found"}
    return {m: os.path.relpath(m, start=base) for m in models}


# region UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h2("Wood Log Defect Segmentation – 3D Viewer"),

        ui.input_file("tiff_file", "Upload .TIFF log", accept=[".tiff", ".tif"]),
        ui.input_select("model_path", "Select Model Checkpoint", list_model_checkpoints()),

        ui.input_checkbox("use_crf", "Apply CRF postprocessing", value=True),
        ui.input_checkbox("use_mrf", "Apply 3D MRF refinement", value=True),

        ui.input_action_button("run_btn", "Run Segmentation", class_="btn-primary"),
        width=300
    ),

    ui.layout_column_wrap(
        output_widget("plot"),
        heights="100vh",
        gap="0"
    )
)
# endregion

# region Server
def server(input, output, session):
    fig_val = reactive.Value(None)

    loop = asyncio.get_event_loop()  # Shiny main event loop

    # The background worker that performs segmentation
    def run_pipeline_background(file_path, model_path, use_crf, use_mrf, prog):
        try:
            pipe = SegmentationPipeline(model_type="cnn", use_crf=use_crf, use_mrf=use_mrf)

            # Called during segmentation per slice
            def progress_callback(current, total):
                pct = int((current / total) * 100)
                msg = f"Segmenting slices ({current}/{total})"

                loop.call_soon_threadsafe(
                    lambda: prog.set(message=msg, value=pct)
                )

            # Run the pipeline
            refined_volume, fig = pipe.run(
                tiff_path=file_path,
                model_path=model_path,
                visualize=True,
                progress_callback=progress_callback
            )

            # Final UI update for mesh display
            loop.call_soon_threadsafe(lambda: fig_val.set(fig))
            loop.call_soon_threadsafe(lambda: prog.set(message="Segmentation complete", value=100))
            loop.call_soon_threadsafe(lambda: prog.close())

        except Exception as e:
            print("Error in background thread:", e)
            loop.call_soon_threadsafe(lambda: prog.set(message=f"Error: {e}"))
            loop.call_soon_threadsafe(lambda: prog.close())

    # Start segmentation when Run button pressed
    @reactive.effect
    @reactive.event(input.run_btn)
    def _():
        fileinfo = input.tiff_file()
        if not fileinfo:
            return

        # Copy file to safe temp path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff")
        shutil.copyfile(fileinfo[0]["datapath"], tmp.name)
        tmp.close()

        # Create progress bar on main thread BEFORE starting worker
        prog = ui.Progress(session=session, min=0, max=100)
        prog.set(message="Starting segmentation...", value=0)

        # Launch worker thread
        t = threading.Thread(
            target=run_pipeline_background,
            args=(tmp.name, input.model_path(), input.use_crf(), input.use_mrf(), prog),
            daemon=True,
        )
        t.start()

    # Render final mesh figure
    @output
    @render_widget
    def plot():
        return fig_val.get()
# endregion

app = App(app_ui, server)
