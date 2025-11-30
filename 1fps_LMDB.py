import os
import lmdb
from tqdm import tqdm

# ----------- CONFIGURACIÓN -----------

src_path = "xai_test_data.lmdb"       # LMDB original (archivo o directorio)
dst_path = "xai_test_data_1fps.lmdb"  # LMDB destino

commit_every = 2000  # commits cada N entradas (evita levantar demasiada RAM)
map_size_gb = 12   # tamaño del nuevo LMDB (ajustable si recibes MapFullError)

# -------------------------------------

def get_label_from_key(key_str):
    """Asigna etiqueta según la ruta que indica el tipo de manipulación."""
    if key_str.startswith("original_sequences"):
        return 0
    if key_str.startswith("manipulated_sequences/NeuralTextures"):
        return 1
    if key_str.startswith("manipulated_sequences/Face2Face"):
        return 2
    if key_str.startswith("manipulated_sequences/Deepfakes"):
        return 3
    if key_str.startswith("manipulated_sequences/FaceSwap"):
        return 4
    return None


def is_dir(path):
    return os.path.isdir(path)


print("Abriendo LMDB origen...")
src_is_dir = is_dir(src_path)
src_env = lmdb.open(src_path, readonly=True, lock=False, subdir=src_is_dir)

dst_is_dir = src_is_dir  # normalmente quieres mantener el mismo formato
dst_map_size = map_size_gb * (1 << 30)

if os.path.exists(dst_path):
    raise FileExistsError(f"El destino '{dst_path}' ya existe. Elimina o usa otro nombre.")

print("Creando LMDB destino...")
dst_env = lmdb.open(dst_path, map_size=dst_map_size, subdir=dst_is_dir)

count_in = 0
count_out = 0
skipped_unrecognized = 0

with src_env.begin(write=False) as src_txn:
    cursor = src_txn.cursor()

    print("Procesando claves...")
    dst_txn = dst_env.begin(write=True)  # transacción inicial

    for key, val in tqdm(cursor, desc="Frames procesados"):
        count_in += 1

        key_str = key.decode("utf-8", errors="ignore")
        filename = key_str.split("/")[-1]
        frame_part = filename.split("_")[0]

        # Extraer número de frame
        try:
            frame_num = int(frame_part)
        except:
            continue

        # Elegir 1 de cada 24
        if (frame_num % 24) != 0:
            continue

        # Etiqueta según tipo
        label = get_label_from_key(key_str)
        if label is None:
            skipped_unrecognized += 1
            continue

        # Valor nuevo = etiqueta + imagen
        new_val = bytes([label]) + val
        dst_txn.put(key, new_val)
        count_out += 1

        # Commit periódico
        if count_out % commit_every == 0:
            dst_txn.commit()
            print(f"Commit #{count_out // commit_every} — almacenados {count_out}")
            dst_txn = dst_env.begin(write=True)

    # commit final
    dst_txn.commit()

print("\n----------- RESUMEN -----------")
print("Entradas leídas:", count_in)
print("Guardadas (1fps):", count_out)
print("Claves sin categoría:", skipped_unrecognized)
print("Nuevo LMDB creado en:", dst_path)
