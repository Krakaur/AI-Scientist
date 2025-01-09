def split_ris_with_metadata(input_file, output_dir, batch_size=10):
    """
    Divide un archivo RIS en múltiples archivos más pequeños con metadatos esenciales.
    Extrae Título (TI), Autor (AU) y Abstract (AB).

    Args:
        input_file (str): Ruta al archivo RIS de entrada.
        output_dir (str): Directorio donde se guardarán los archivos divididos.
        batch_size (int): Cantidad de referencias por archivo.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_batch = []
    current_reference = []
    batch_count = 0
    reference_count = 0

    for line in lines:
        current_reference.append(line)
        if line.startswith("ER"):  # Fin de una referencia en RIS
            current_batch.append("".join(current_reference))
            current_reference = []
            reference_count += 1

            if reference_count >= batch_size:
                output_file = os.path.join(output_dir, f'batch_{batch_count}.ris')
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(current_batch)
                current_batch = []
                batch_count += 1
                reference_count = 0

    # Guardar el último lote si queda algo
    if current_batch:
        output_file = os.path.join(output_dir, f'batch_{batch_count}.ris')
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.writelines(current_batch)


def extract_metadata_from_ris(ris_file):
    """
    Extrae metadatos esenciales (TI, AU, AB) de un archivo RIS.

    Args:
        ris_file (str): Ruta al archivo RIS.
    
    Returns:
        list[dict]: Lista de referencias con los campos extraídos.
    """
    references = []
    current_reference = {}
    with open(ris_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("TI  -"):  # Título
                current_reference["title"] = line[6:].strip()
            elif line.startswith("AU  -"):  # Autor
                current_reference.setdefault("authors", []).append(line[6:].strip())
            elif line.startswith("AB  -"):  # Abstract
                current_reference["abstract"] = line[6:].strip()
            elif line.startswith("ER"):  # Fin de referencia
                if current_reference:  # Si se encontró una referencia completa
                    references.append(current_reference)
                    current_reference = {}
    return references


# Dividir el archivo RIS en lotes
split_ris_with_metadata('templates/references.ris', 'output_batches', batch_size=10)

# Probar extracción de metadatos de un lote
batch_file = "output_batches/batch_0.ris"
references = extract_metadata_from_ris(batch_file)
for ref in references:
    print("Título:", ref.get("title", "N/A"))
    print("Autores:", ref.get("authors", []))
    print("Abstract:", ref.get("abstract", "N/A"))
    print("-" * 40)
import os
import json

def save_references_to_files(references, output_dir, references_per_file):
    """
    Guarda los metadatos extraídos en archivos con un número definido de referencias por archivo.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dividir las referencias en lotes
    total_references = len(references)
    for i in range(0, total_references, references_per_file):
        batch = references[i:i + references_per_file]
        file_name = os.path.join(output_dir, f"batch_{i // references_per_file + 1}.json")
        
        with open(file_name, "w") as f:
            json.dump(batch, f, indent=4)

        print(f"Lote guardado en: {file_name}")

if __name__ == "__main__":
    # Usamos las funciones existentes para procesar y extraer datos
    references = extract_metadata_from_ris("templates/references.ris")
    
    # Preguntar cuántas referencias quieres por archivo
    while True:
        try:
            references_per_file = int(input("¿Cuántas referencias deseas por archivo? "))
            if references_per_file > 0:
                break
            else:
                print("El número debe ser mayor que 0.")
        except ValueError:
            print("Por favor, introduce un número válido.")

    # Guardar las referencias en lotes
    save_references_to_files(references, "output_references", references_per_file)
