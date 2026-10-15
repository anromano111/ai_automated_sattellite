SOURCE=/data/img.tif
DEST=/data/giz_aug
SHAPE=/data/vilankulo_all.geojsonl.json

python data_augment.py \
  -i ${SOURCE} \
  -j ${SHAPE} \
  -o ${DEST} \
  -n 600 \
  # -d True

# Train/val split
cd "${DEST}/images"
mkdir train && mkdir val

# Copy last 50 images to val. 52 because of train/val folders
ls "${DEST}/images" | tail -52 | while read -r file; do
    # Ensure it's a file, not a folder
    if [ -f "$DEST/images/$file" ]; then
        mv "$DEST/images/$file" "$DEST/images/val"
    fi
done

# Copy the rest to train.
ls "${DEST}/images" | while read -r file; do
    # Ensure it's a file, not a folder
    if [ -f "$DEST/images/$file" ]; then
        mv "$DEST/images/$file" "$DEST/images/train"
    fi
done

# Train/val split
cd "${DEST}/masks"
mkdir train && mkdir val

# Copy last 50 masks to val. 52 because of train/val folders
ls "${DEST}/masks" | tail -52 | while read -r file; do
    # Ensure it's a file, not a folder
    if [ -f "$DEST/masks/$file" ]; then
        mv "$DEST/masks/$file" "$DEST/masks/val"
    fi
done

# Copy the rest to train.
ls "${DEST}/masks" | while read -r file; do
    # Ensure it's a file, not a folder
    if [ -f "$DEST/masks/$file" ]; then
        mv "$DEST/masks/$file" "$DEST/masks/train"
    fi
done

# Train/val split
cd "${DEST}/labels"
mkdir train && mkdir val

# Copy last 50 labels to val. 52 because of train/val folders
ls "${DEST}/labels" | tail -52 | while read -r file; do
    # Ensure it's a file, not a folder
    if [ -f "$DEST/labels/$file" ]; then
        mv "$DEST/labels/$file" "$DEST/labels/val"
    fi
done

# Copy the rest to train.
ls "${DEST}/labels" | while read -r file; do
    # Ensure it's a file, not a folder
    if [ -f "$DEST/labels/$file" ]; then
        mv "$DEST/labels/$file" "$DEST/labels/train"
    fi
done

echo "Successfully generated YOLO dataset "
