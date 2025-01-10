import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime
from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS


from transformers import pipeline
"""from ris_utils import load_reference"""
from ris_utils import extract_metadata_from_ris

# Cargar el token desde config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Establecer el token de Hugging Face como variable de entorno
os.environ["HF_API_TOKEN"] = config.get("hf_api_token")

NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluar referencias bibliográficas")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Modelo LLM para evaluar referencias.",
    )
    return parser.parse_args()

def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def worker(
        queue,
        base_dir,
        results_dir,
        model,
        client,
        client_model,
        writeup,
        improvement,
        gpu_id,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            model,
            client,
            client_model,
            writeup,
            improvement,
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        client,
        client_model,
        writeup,
        improvement,
        log_file=False,
):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
    except Exception as e:
        print(f"Error: {e}")

        print_time()

from transformers import pipeline

def truncate_abstract(abstract, max_tokens=200):
    """
    Trunca un abstract largo en fragmentos más pequeños manejables.
    Args:
        abstract (str): El texto del abstract.
        max_tokens (int): Número máximo de tokens permitidos por fragmento.
    Returns:
        list: Fragmentos del abstract truncado.
    """
    # Dividimos el abstract en palabras/tokens
    tokens = abstract.split()
    fragments = []

    for i in range(0, len(tokens), max_tokens):
        fragments.append(" ".join(tokens[i:i + max_tokens]))

    return fragments




if __name__ == "__main__":
    # Crear pipeline Hugging Face
    generator = pipeline("text-generation", model="gpt2")  # Sustituye "gpt2" por el modelo que prefieras
    
    # Ruta a la carpeta de archivos JSON generados
    references_dir = "output_references"
    json_files = [osp.join(references_dir, f) for f in os.listdir(references_dir) if f.endswith(".json")]

   # Procesar cada archivo JSON con referencias
ideas = []  # Inicializa una lista para almacenar los resultados
for json_file in json_files:
    with open(json_file, "r") as f:
        references = json.load(f)

    for reference in references:
        if not isinstance(reference, dict):
            print(f"Referencia en formato inesperado: {reference}")
            continue

        # Extraer información básica
        title = reference.get("title", "Título no disponible")
        authors = ", ".join(reference.get("authors", []))
        abstract = reference.get("abstract", "Abstracto no disponible")

        print(f"\nProcesando referencia: {title}")
        print(f"Autores: {authors}")

        if abstract:
            print(f"Abstract:\n{abstract[:200]}... (truncado para mostrar solo los primeros 200 caracteres)\n")

            # Fragmentar el abstract si es demasiado largo
            abstract_fragments = truncate_abstract(abstract)

            generated_text = ""
            for fragment in abstract_fragments:
                try:
                    result = generator(fragment, max_new_tokens=50, num_return_sequences=1)
                    generated_text += result[0]["generated_text"] + " "
                except Exception as e:
                    print(f"Error al procesar el fragmento: {e}")

            print(f"Texto generado por el modelo GPT-2:\n{generated_text.strip()}\n")

            # Almacenar resultados
            ideas.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "generated_text": generated_text.strip()
            })

# Guardar resultados
output_file = osp.join(references_dir, "ideas.json")
with open(output_file, "w") as f:
    json.dump(ideas, f, indent=4)
print(f"\nResultados guardados en: {output_file}")

