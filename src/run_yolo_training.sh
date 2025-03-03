for dataset in datasets/*/; do
    index_tag=$(basename "$dataset" | sed -E 's/^([a-zA-Z-]+)[0-9]*.*/\1/' | sed 's/-$//')
    testing_tag=$(basename "$dataset" | sed -E 's/^[^_]+_([^_]+).*/\1/' | sed 's/[0-9]*$//')
    echo python -m detection.yolo -d "$dataset" -n yolo --tag "$index_tag index" --tag "$testing_tag ds testing" --tag "YOLOv10n"
    python -m detection.yolo -d "$dataset" -n yolo --tag "$index_tag index" --tag "$testing_tag ds testing" --tag "YOLOv10n"
done