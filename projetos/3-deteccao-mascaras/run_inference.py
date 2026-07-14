import os

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.pt) usando YOLO(..., task="detect")
#   2. Rodar inferência em pelo menos 5 imagens do conjunto de validação
#      (dataset/images/val/), UMA DE CADA VEZ — o model.tflite exportado
#      aceita apenas 1 imagem por chamada (batch=1), que é aliás o cenário
#      real de uso em edge (uma foto por vez, não em lote)
#   3. Imprimir no terminal, para cada amostra, o número de detecções
#   4. As imagens anotadas com as caixas preditas são salvas automaticamente
#      pelo Ultralytics em runs/detect/... — abra essa pasta localmente para
#      conferir visualmente as predições (não precisa ser commitada)
# ---------------------------------------------------------------------------

N_SAMPLES = 5


def main():
    model = YOLO("model.tflite", task="detect")

    val_dir = "dataset/images/val"
    sample_images = sorted(os.listdir(val_dir))[:N_SAMPLES]
    sample_paths = [os.path.join(val_dir, f) for f in sample_images]

    print(f"Rodando inferência em {len(sample_paths)} amostras usando model.tflite:\n")
    for path in sample_paths:
        result = model.predict(
            source=path, save=True, project="inferencia_exemplos", name="predicoes",
            exist_ok=True, verbose=False,
        )[0]
        print(f"{os.path.basename(path)}: {len(result.boxes)} detecção(ões)")

    print("\nImagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/")


if __name__ == "__main__":
    main()
