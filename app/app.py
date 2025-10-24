import sys, os, tempfile, shutil, threading, glob, queue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
from src.pipelines.segmentation_pipeline import SegmentationPipeline

def list_model_checkpoints():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    models = glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
    if not models:
        return {"": "No checkpoints found"}
    return {m: os.path.relpath(m, start=base) for m in models}


# ---------------- UI ----------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h2("Wood Log Defect Segmentation – 3D Viewer"),
        ui.input_file("tiff_file", "Upload .TIFF log", accept=[".tiff", ".tif"]),
        ui.input_select("model_path", "Select Model Checkpoint", {"../logs/old_model/checkpoints/best.pt": "old_model/best.pt",
                                                                                    "../logs/test_model/checkpoints/best.pt": "test_model/best.pt"} ),
        ui.input_checkbox("use_crf", "Apply CRF postprocessing", value=True),
        ui.input_checkbox("use_mrf", "Apply 3D MRF refinement", value=True),
        ui.input_action_button("run_btn", "Run Segmentation", class_="btn-primary"),
        ui.hr(),
        ui.output_ui("progress_ui"),
    ),
    ui.layout_column_wrap(
        ui.card(
            ui.h4("3D Mesh Segmentation"),
            output_widget("plot", height="90vh"),
            full_screen=True,
            fill=True,
            style="min-height: 80vh;"
        ),
        fill=True
    )
)



# ---------------- Server ----------------
import queue
import tempfile, shutil, threading
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
from src.pipelines.segmentation_pipeline import SegmentationPipeline

def server(input, output, session):
    running = reactive.Value(False)
    status = reactive.Value("Ready to run.")
    fig_val = reactive.Value(None)
    update_queue = queue.Queue()

    # --- Background thread (run segmentation) ---
    def run_pipeline_background(file_path, model_path, use_crf, use_mrf):
        try:
            pipe = SegmentationPipeline(model_type="cnn", use_crf=use_crf, use_mrf=use_mrf)
            print("Starting segmentation...")

            refined_volume, fig = pipe.run(
                tiff_path=file_path,
                model_path=model_path,
                visualize=True,
            )

            print("Pipeline finished.")
            update_queue.put(("done", fig))
        except Exception as e:
            print("Error in background thread:", e)
            update_queue.put(("error", str(e)))

    # --- Reactive poller for thread results ---
    @reactive.effect
    def poll_queue():
        reactive.invalidate_later(1.0)  # tick every second
        if not update_queue.empty():
            msg, data = update_queue.get()
            if msg == "done":
                print("Queue got: done → updating UI.")
                status.set("Segmentation completed.")
                fig_val.set(data)
            elif msg == "error":
                status.set(f"Error: {data}")
            running.set(False)

    # --- Start on button click ---
    @reactive.effect
    @reactive.event(input.run_btn)
    def _():
        fileinfo = input.tiff_file()
        if not fileinfo:
            status.set("No TIFF uploaded.")
            return

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff")
        shutil.copyfile(fileinfo[0]["datapath"], tmp.name)
        tmp.close()

        running.set(True)
        fig_val.set(None)
        status.set("Starting segmentation...")

        threading.Thread(
            target=run_pipeline_background,
            args=(tmp.name, input.model_path(), input.use_crf(), input.use_mrf()),
            daemon=True,
        ).start()

    # --- Status text ---
    @output
    @render.ui
    def progress_ui():
        return ui.p(status.get(), style="font-size: 0.95rem; color: #333;")

    # --- Plotly 3D figure output ---
    @output
    @render_widget
    def plot():
        return fig_val.get()




app = App(app_ui, server)
