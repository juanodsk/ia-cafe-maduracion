"""Build the Spanish training report from saved validation and CV results.

The script never opens a test prediction CSV, a test metrics file, or the
contents of evaluacion_final.json. Figures and tables use recorded results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "experimentos" / "transfer_01"
MODEL_NAMES = ("resnet50", "efficientnet_b0", "densenet121", "vit_b_16", "swin_t")
MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet-121",
    "vit_b_16": "ViT-B/16",
    "swin_t": "Swin-T",
}
ENSEMBLE_DISPLAY = {
    "heterogeneo_cinco": "Heterogéneo de cinco arquitecturas",
    "familia_cnn": "Promedio de las tres CNN",
    "familia_transformer": "Promedio de los dos Transformers",
    "hibrido_familias": "Híbrido con igual peso por familia",
    "homogeneo_cinco_folds": "Homogéneo de clasificadores por fold",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _format(value: Any, decimals: int = 4, *, percentage: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "Pendiente"
    if percentage:
        return f"{number * 100:.{decimals}f}".replace(".", ",") + " %"
    return f"{number:.{decimals}f}".replace(".", ",")


def _integer(value: Any) -> str:
    number = _number(value)
    return f"{int(number):,}".replace(",", ".") if number is not None else "Pendiente"


def _text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _link(path: Path, report_dir: Path, label: str) -> str:
    relative = Path(os.path.relpath(path, report_dir)).as_posix()
    return f"[{label}](<{relative}>)"


def _first_json(paths: list[Path]) -> tuple[dict[str, Any], Path | None]:
    for path in paths:
        if path.is_file():
            return _read_json(path), path
    return {}, None


def _read_history(model_dir: Path) -> list[dict[str, float]]:
    for path in (model_dir / "history.csv", model_dir / "final_fit" / "history.csv"):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            values = list(csv.DictReader(handle))
        history = []
        for row in values:
            epoch = _number(row.get("epoch"))
            loss = _number(row.get("train_loss"))
            f1 = _number(row.get("valid_macro_f1"))
            if epoch is not None and loss is not None and f1 is not None:
                history.append({"epoch": epoch, "train_loss": loss, "valid_macro_f1": f1})
        return history
    return []


def _collect_models(run_dir: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    collected = {}
    for name in names:
        model_dir = run_dir / name
        result, result_path = _first_json([model_dir / "resultado.json", model_dir / "study_summary.json"])
        validation = result.get("validation_metrics", {})
        if not validation:
            validation, _ = _first_json([
                model_dir / "validacion" / "metrics.json", model_dir / "validation" / "metrics.json"
            ])
        cv = result.get("cv_summary") or _read_json(model_dir / "cross_validation" / "summary.json")
        hyperband = result.get("hyperband") or _read_json(model_dir / "hyperband" / "best_params.json")
        spec = result.get("model_spec") or validation.get("metadata", {}).get("model_spec")
        if not spec:
            feature_meta = _read_json(run_dir / "features" / name / "train.json")
            spec = feature_meta.get("configuration", {}).get("model", {})
        collected[name] = {
            "result": result,
            "result_path": result_path,
            "validation": validation,
            "cv": cv,
            "hyperband": hyperband,
            "spec": spec or {},
            "history": _read_history(model_dir),
            "model_dir": model_dir,
        }
    return collected


def _plot_comparison(run_dir: Path, models: dict[str, dict[str, Any]]) -> Path | None:
    available = [name for name, row in models.items() if _number(row["validation"].get("macro_f1")) is not None]
    if not available:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator

    figure, axis = plt.subplots(figsize=(10.8, 6.0), layout="constrained")
    positions = list(range(len(available)))
    width = 0.35
    validation = [float(models[name]["validation"]["macro_f1"]) for name in available]
    axis.bar([x - width / 2 for x in positions], validation, width, color="#2463a6", label="Validación fija")
    cv_positions, cv_means, cv_stds = [], [], []
    for x, name in zip(positions, available):
        mean = _number(models[name]["cv"].get("macro_f1_mean"))
        std = _number(models[name]["cv"].get("macro_f1_std"))
        if mean is not None and std is not None:
            cv_positions.append(x + width / 2)
            cv_means.append(mean)
            cv_stds.append(std)
        else:
            axis.text(x + width / 2, 0.04, "CV pendiente", rotation=90, fontsize=8, ha="center", va="bottom")
    if cv_positions:
        axis.bar(cv_positions, cv_means, width, yerr=cv_stds, capsize=4, color="#d28035", label="CV en train: media ± DE")
    axis.set_xticks(positions, [MODEL_DISPLAY.get(name, name) for name in available])
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1 macro")
    axis.set_title("Clasificación de madurez de café\nValidación fija y validación cruzada por grupos")
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    figure_dir = run_dir / "figuras"
    figure_dir.mkdir(parents=True, exist_ok=True)
    target = figure_dir / "comparacion_validacion.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _plot_histories(run_dir: Path, models: dict[str, dict[str, Any]]) -> Path | None:
    if not any(row["history"] for row in models.values()):
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), layout="constrained")
    for axis, (name, row) in zip(axes.flat, models.items()):
        history = row["history"]
        axis.set_title(MODEL_DISPLAY.get(name, name))
        axis.set_xlabel("Época de entrenamiento de la cabeza")
        if not history:
            axis.text(0.5, 0.5, "Historial pendiente", ha="center", va="center", transform=axis.transAxes)
            axis.set_yticks([])
            axis.set_xticks([])
            continue
        epochs = [item["epoch"] for item in history]
        loss_line = axis.plot(epochs, [item["train_loss"] for item in history], color="#d28035", label="Pérdida train")
        axis.set_ylabel("Pérdida train", color="#a86220")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        second = axis.twinx()
        f1_line = second.plot(epochs, [item["valid_macro_f1"] for item in history], color="#2463a6", label="F1 macro valid")
        second.set_ylabel("F1 macro valid", color="#2463a6")
        second.set_ylim(0, 1)
        best_epoch = _number(row["result"].get("best_epoch", row["hyperband"].get("best_epoch")))
        if best_epoch is not None:
            axis.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1)
        lines = loss_line + f1_line
        axis.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    for axis in list(axes.flat)[len(models):]:
        axis.axis("off")
    figure.suptitle("Curvas del ajuste seleccionado\nExtractores ImageNet congelados y clasificadores entrenados", fontsize=14)
    figure_dir = run_dir / "figuras"
    figure_dir.mkdir(parents=True, exist_ok=True)
    target = figure_dir / "curvas_aprendizaje.png"
    figure.savefig(target, dpi=170)
    plt.close(figure)
    return target


def generate_report(run_dir: str | Path = DEFAULT_RUN, output_path: str | Path | None = None) -> dict[str, Any]:
    """Generate a validation-only report without importing training modules."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {run_dir}")
    output_path = Path(output_path).resolve() if output_path else ROOT / "docs" / f"resultados_{run_dir.name}.md"
    report_dir = output_path.parent
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    config = _read_json(run_dir / "configuracion.json")
    state = _read_json(run_dir / "estado.json")
    selection = _read_json(run_dir / "seleccion_validacion.json")
    configured_names = config.get("models", list(MODEL_NAMES))
    if not isinstance(configured_names, list) or not configured_names or any(name not in MODEL_NAMES for name in configured_names):
        raise ValueError("Experiment models do not match the five supported architectures.")
    names = list(dict.fromkeys(configured_names))
    models = _collect_models(run_dir, names)
    completed = [name for name in names if _number(models[name]["validation"].get("macro_f1")) is not None]
    comparison = _plot_comparison(run_dir, models)
    histories = _plot_histories(run_dir, models)
    final_path = run_dir / "evaluacion_final.json"
    final_exists = final_path.is_file()  # Deliberately never open this test artifact.
    lines = [
        "# Resultados de clasificación de madurez de café",
        "",
        f"El experimento `{_text(run_dir.name)}` registra resultados de validación para {len(completed)} de {len(names)} arquitecturas configuradas. "
        "Las métricas siguientes proceden de los artefactos guardados; los modelos pendientes se indican como tales.",
        "",
        "## Alcance del entrenamiento",
        "",
        "Se utilizan extractores preentrenados en ImageNet con sus pesos congelados. Se extraen características de cada fruto por lotes y se entrena "
        "una cabeza de clasificación para las cinco etapas de madurez. La normalización de características y los pesos de clase se calculan exclusivamente "
        "con las filas de entrenamiento de cada ajuste. Esta fase no actualiza todas las capas de las CNN o los Transformers.",
        "",
        "Cada recorte hereda el conjunto y grupo de su imagen padre. La división fija es 70 % / 15 % / 15 % sobre imágenes exportadas únicas, "
        "con redondeo entero. El F1 macro de validación selecciona hiperparámetros, época y candidatos al ensemble; accuracy y balanced accuracy "
        "se presentan como medidas complementarias.",
        "",
        "| Parámetro del experimento | Valor registrado |",
        "| --- | --- |",
        f"| Semilla general | {_integer(config.get('seed'))} |",
        f"| Dispositivo | {_text(config.get('gpu') or config.get('device') or 'No registrado')} |",
        f"| Ensayos solicitados por arquitectura | {_integer(config.get('n_trials'))} |",
        f"| Máximo de épocas por ensayo | {_integer(config.get('max_epochs'))} |",
        f"| Folds solicitados | {_integer(config.get('cv_folds'))} |",
        f"| Fase del runner | `{_text(state.get('stage', 'No registrada'))}` |",
        "",
        "## Arquitecturas",
        "",
        "| Arquitectura | Familia | Pesos de origen | Dimensión de características | Frutos de train / valid |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for name in names:
        row = models[name]
        spec = row["spec"]
        result = row["result"]
        family = "CNN" if name in MODEL_NAMES[:3] else "Transformer"
        crops = _integer(result.get("training_crops")) + " / " + _integer(result.get("validation_crops"))
        lines.append(f"| {MODEL_DISPLAY[name]} | {family} | `{_text(spec.get('weights', 'Pendiente'))}` | {_integer(spec.get('feature_dim', result.get('feature_dim')))} | {crops} |")
    lines.extend([
        "",
        "Los tamaños de entrada, interpolación y normalización corresponden al preset oficial de los pesos seleccionado para cada arquitectura. "
        "La extracción de características utiliza ese preprocesamiento determinista, sin aumentos aleatorios nuevos.",
        "",
        "## Resultados de validación",
        "",
        "| Arquitectura | Accuracy | Balanced accuracy | F1 macro | F1 ponderado | F1 macro CV media ± DE | Época elegida |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    ordered = sorted(names, key=lambda name: _number(models[name]["validation"].get("macro_f1")) if _number(models[name]["validation"].get("macro_f1")) is not None else -1, reverse=True)
    for name in ordered:
        row = models[name]
        metrics, cv = row["validation"], row["cv"]
        cv_text = "Pendiente"
        if _number(cv.get("macro_f1_mean")) is not None and _number(cv.get("macro_f1_std")) is not None:
            cv_text = _format(cv["macro_f1_mean"]) + " ± " + _format(cv["macro_f1_std"])
        epoch = row["result"].get("best_epoch", row["hyperband"].get("best_epoch"))
        lines.append(f"| {MODEL_DISPLAY[name]} | {_format(metrics.get('accuracy'), 2, percentage=True)} | {_format(metrics.get('balanced_accuracy'), 2, percentage=True)} | {_format(metrics.get('macro_f1'))} | {_format(metrics.get('weighted_f1'))} | {cv_text} | {_integer(epoch)} |")
    if comparison:
        lines.extend(["", f"![Comparación de F1 macro en validación y CV]({comparison.as_posix()})"])
    lines.extend([
        "",
        "La validación cruzada utiliza `StratifiedGroupKFold` dentro de train: cada grupo completo se mantiene en un solo fold. "
        "Los hiperparámetros y el presupuesto de épocas se fijan con la validación separada y se reutilizan en los folds. "
        "Por ello, la media y la desviación estándar describen una CV condicionada a esa selección, no una búsqueda anidada. "
        "La desviación estándar es muestral entre folds; no es un intervalo de confianza ni variación entre cinco semillas completas del backbone.",
        "",
        "## Hiperparámetros seleccionados",
        "",
        "Hyperband interrumpe ensayos menos prometedores y prioriza recursos por épocas. En esta fase los hiperparámetros optimizados pertenecen "
        "a la cabeza de clasificación: tasa de aprendizaje, tamaño de lote, weight decay, dropout, dimensión oculta y ponderación de clases.",
        "",
        "| Arquitectura | Learning rate | Batch | Weight decay | Dropout | Dimensión oculta | Potencia de peso de clase | Ensayos completos / podados |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name in names:
        row = models[name]
        hyperband = row["hyperband"]
        params = row["result"].get("best_params") or hyperband.get("best_params", {})
        trial_counts = _integer(hyperband.get("completed_trials")) + " / " + _integer(hyperband.get("pruned_trials"))
        lr = "Pendiente" if _number(params.get("lr")) is None else f"{float(params['lr']):.6g}"
        decay = "Pendiente" if _number(params.get("weight_decay")) is None else f"{float(params['weight_decay']):.6g}"
        lines.append(f"| {MODEL_DISPLAY[name]} | {lr} | {_integer(params.get('batch_size'))} | {decay} | {_format(params.get('dropout'), 2)} | {_integer(params.get('hidden_dim'))} | {_format(params.get('class_weight_power'), 2)} | {trial_counts} |")
    lines.extend(["", "## Curvas de aprendizaje", ""])
    if histories:
        lines.append(f"![Pérdida de entrenamiento y F1 macro de validación por arquitectura]({histories.as_posix()})")
        lines.extend(["", "Cada panel muestra el historial real del ajuste que reproduce el ensayo seleccionado. La línea vertical marca la época del checkpoint elegido. "
                      "La pérdida de train y el F1 macro de validación utilizan ejes y unidades distintos."])
    else:
        lines.append("Todavía no hay historiales de ajustes seleccionados guardados para construir las curvas.")
    lines.extend([
        "",
        "## Ensembles y selección",
        "",
        "El ensemble heterogéneo de cinco arquitecturas promedia sus probabilidades con igual peso por modelo. Los promedios de familia CNN y "
        "Transformer combinan arquitecturas distintas dentro de esa familia; pertenecer a la misma familia no los convierte en ensembles homogéneos.",
        "",
        "El híbrido de familias asigna la mitad del peso total a las CNN y la otra mitad a los Transformers. Es un promedio jerárquico de "
        "probabilidades, equivalente a pesos 1/6 para cada CNN y 1/4 para cada Transformer; no es una nueva red de fusión entrenada.",
        "",
        "El ensemble homogéneo promedia las cabezas de la mejor arquitectura individual, elegida por validación, entrenadas en los folds. "
        "Estas cabezas comparten el extractor ImageNet congelado y difieren en sus subconjuntos de entrenamiento.",
        "",
    ])
    definitions = selection.get("ensemble_definitions", {})
    ensemble_rows = []
    for name in ENSEMBLE_DISPLAY:
        if name not in definitions:
            continue
        metrics, metrics_path = _first_json([
            run_dir / "ensembles" / name / "validacion" / "metrics.json",
            run_dir / "ensembles" / name / "validation" / "metrics.json",
        ])
        if not metrics:
            continue
        definition = definitions[name]
        components = ", ".join(MODEL_DISPLAY.get(item, item) for item in definition.get("models", []))
        if definition.get("cv_checkpoints"):
            components += f"; {len(definition['cv_checkpoints'])} cabezas por fold"
        ensemble_rows.append((name, metrics, components, metrics_path))
    if ensemble_rows:
        lines.extend([
            "| Ensemble | Componentes | F1 macro valid | Accuracy valid |",
            "| --- | --- | ---: | ---: |",
        ])
        for name, metrics, components, _ in ensemble_rows:
            lines.append(f"| {ENSEMBLE_DISPLAY[name]} | {_text(components)} | {_format(metrics.get('macro_f1'))} | {_format(metrics.get('accuracy'), 2, percentage=True)} |")
    else:
        lines.append("Todavía no hay resultados guardados de ensembles de validación.")
    selected = selection.get("selected_candidate")
    if selected:
        display = MODEL_DISPLAY.get(selected, ENSEMBLE_DISPLAY.get(selected, selected))
        score = selection.get("scores", {}).get(selected)
        lines.extend(["", f"Candidato seleccionado por validación: **{_text(display)}**, con F1 macro {_format(score)}. "
                      "Esta selección precede a la apertura de test y no se basa en sus resultados."])
    lines.extend(["", "## Conjunto de prueba", ""])
    if final_exists:
        lines.append("Existe un artefacto de evaluación final: " + _link(final_path, report_dir, "evaluacion_final.json") + ". "
                     "Este informe utiliza exclusivamente resultados de validación y CV; el generador no abre ese archivo ni lee métricas o predicciones de prueba.")
    elif state.get("stage") == "final_test_evaluation":
        lines.append("El runner registra una evaluación final en curso, posterior al bloqueo de la selección por validación. "
                     "Este informe no incorpora ni lee resultados de prueba.")
    else:
        lines.append("El conjunto de prueba permanece reservado para una evaluación final después de fijar modelos, hiperparámetros y ensembles. "
                     "Las tablas y figuras anteriores no utilizan predicciones ni métricas de test.")
    lines.extend([
        "",
        "## Artefactos de validación y reproducibilidad",
        "",
    ])
    for name in names:
        row = models[name]
        artifacts = []
        if row["result_path"]:
            artifacts.append(_link(row["result_path"], report_dir, "resultado"))
        for subdir in ("validacion", "validation"):
            matrix = row["model_dir"] / subdir / "confusion_matrix.png"
            if matrix.is_file():
                artifacts.append(_link(matrix, report_dir, "matriz de confusión"))
                predictions = matrix.parent / "predictions.csv"
                if predictions.is_file():
                    artifacts.append(_link(predictions, report_dir, "predicciones de validación"))
                break
        cv_path = row["model_dir"] / "cross_validation" / "fold_metrics.csv"
        if cv_path.is_file():
            artifacts.append(_link(cv_path, report_dir, "métricas por fold"))
        lines.append(f"- **{MODEL_DISPLAY[name]}**: " + (", ".join(artifacts) if artifacts else "artefactos pendientes") + ".")
    for name, _, _, metrics_path in ensemble_rows:
        matrix = metrics_path.parent / "confusion_matrix.png"
        if matrix.is_file():
            lines.append(f"- **{ENSEMBLE_DISPLAY[name]}**: " + _link(matrix, report_dir, "matriz de confusión de validación") + ".")
    if config.get("dataset_manifest_sha256"):
        lines.extend(["", f"SHA256 del manifiesto de imágenes: `{_text(config['dataset_manifest_sha256'])}`."])
    if config.get("annotation_manifest_sha256"):
        lines.extend(["", f"SHA256 del manifiesto de anotaciones: `{_text(config['annotation_manifest_sha256'])}`."])
    lines.extend([
        "",
        "## Límites y trabajo pendiente",
        "",
        "La descarga v5 contiene aumentos previos de cizallamiento y exposición; conservar los grupos evita compartir las variantes identificadas "
        "entre conjuntos, pero no recupera originales sin aumentos. Tampoco hay identificadores de finca, planta o fruto físico que permitan "
        "garantizar independencia entre todas las capturas. El desbalance de clases se aborda con pesos calculados en train y se hace visible "
        "mediante F1 macro y las matrices de confusión.",
        "",
        "Estos resultados, cuando estén completos, corresponden a aprendizaje por transferencia con extractores congelados y cabezas aprendidas. "
        "El ajuste de todas las capas del backbone sería una fase experimental distinta y debe registrarse por separado.",
        "",
        "Las extensiones siguientes utilizan únicamente entrenamiento y validación. Sus presupuestos son pilotos distintos; "
        "no constituyen una comparación con igual cómputo frente al estudio centralizado.",
        "",
        "## Experimentos GAN federado y distribuido",
        "",
    ])
    extension_file = run_dir / "resumen_extensiones.json"
    if extension_file.is_file():
        extension_data = json.loads(extension_file.read_text(encoding="utf-8"))
        phases = extension_data.get("phases", {})
        lines.extend(["| Experimento | Estado | Macro F1 de validación |", "| --- | --- | ---: |"])
        for key in ("gan", "gan_ablation", "federated", "distributed"):
            phase = phases.get(key, {})
            result = phase.get("result", {})
            value = result.get("validation_metrics", {}).get("macro_f1")
            score = f"{value:.4f}" if isinstance(value, (float, int)) else "No aplica" if key == "gan" else "Pendiente"
            lines.append(f"| {key} | {phase.get('status', 'pending')} | {score} |")
        gan = phases.get("gan", {}).get("result", {})
        if gan.get("synthetic_images"):
            lines.extend(["", f"La GAN generó {gan['synthetic_images']} imágenes de las cuatro clases minoritarias, restringidas a train. "
                          "El piloto no garantiza calidad agronómica de las etiquetas sintéticas."])
        visual_review = run_dir / "gan_revision_visual.json"
        if visual_review.is_file():
            review = json.loads(visual_review.read_text(encoding="utf-8-sig"))
            lines.extend(["", "**Revisión visual de la GAN:** " + review["finding"] + " " + review["use"]])
        ablation = phases.get("gan_ablation", {}).get("result", {})
        if "macro_f1_change" in ablation:
            lines.extend(["", f"La ablación con sintéticos cambió macro F1 en {ablation['macro_f1_change']:+.4f} frente al mismo modelo base, "
                          "con la misma semilla, parámetros y presupuesto de épocas. No se incorpora la GAN global a los folds de CV."])
        lines.extend(["", "FedAvg simula tres clientes con grupos separados en un equipo. DDP utiliza dos procesos CPU reales con Gloo en ese mismo equipo; "
                      "no es un despliegue en varias fincas ni una prueba de escalabilidad entre máquinas.", "",
                      _link(extension_file, report_dir, "Resumen trazable de extensiones") + "."])
    else:
        lines.append("Las extensiones aún no tienen resultados registrados.")
    lines.extend([
        "",
        "## Referencias técnicas",
        "",
    ])
    references = {}
    for name in names:
        reference = models[name]["spec"].get("reference")
        if reference:
            references[MODEL_DISPLAY[name]] = reference
    for label, url in references.items():
        lines.append(f"- [{label} en TorchVision]({url}).")
    lines.extend([
        "- [HyperbandPruner de Optuna](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html).",
        "- [StratifiedGroupKFold de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).",
        "",
    ])
    report_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output_path)
    return {
        "report": str(output_path), "models_with_validation": completed,
        "models_pending_validation": [name for name in names if name not in completed],
        "comparison_figure": str(comparison) if comparison else None,
        "learning_curves": str(histories) if histories else None,
        "test_metrics_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
