import numpy as np
import keras_ocr
import tensorflow as tf
import re
from PIL import Image
import io

def decode_prediction(pred, num_to_char):
    pred_decoded = tf.keras.backend.ctc_decode(
        pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True
    )[0][0].numpy()
    result = ""
    for idx in pred_decoded[0]:
        if idx == 0:
            continue
        result += num_to_char.get(idx-1, "")
    return result

def crop_and_predict_words(image_np, recognizer, num_to_char, detector, IMG_HEIGHT, IMG_WIDTH):
    image = image_np
    prediction_groups = detector.detect([image])
    boxes = prediction_groups[0]
    crops = []
    for box in boxes:
        cropped = keras_ocr.tools.warpBox(image, box, target_height=IMG_HEIGHT, target_width=IMG_WIDTH)
        cropped = tf.image.rgb_to_grayscale(cropped)
        cropped = tf.image.convert_image_dtype(cropped, tf.float32)
        crops.append(cropped)
    if not crops:
        return []
    batch_size = 16
    results = []
    for i in range(0, len(crops), batch_size):
        batch = tf.stack(crops[i:i+batch_size], axis=0)
        preds = recognizer.model.predict(batch, verbose=0)
        for j in range(preds.shape[0]):
            hasil = decode_prediction(preds[j][None, ...], num_to_char)
            results.append(hasil)
    return results

def extract_nutrition_json(texts):
    labels = {
        "energi": ["energi", "energi total", "energy"],
        "protein": ["protein"],
        "lemak total": ["lemak total", "lemak", "total fat", "fat"],
        "karbohidrat": ["karbohidrat", "karbohidrat total", "karbo", "carbohydrate", "carbohydrate total"],
        "serat": ["serat", "fiber"],
        "gula": ["gula", "sugar"],
        "garam": ["garam", "garam (natrium)", "natrium", "sodium", "salt"]
    }
    result = {k: 0 for k in labels}
    for i, line in enumerate(texts):
        l = line.lower()
        for key, keys in labels.items():
            for label in keys:
                if label in l:
                    if key == "energi" and ("kebutuhan energi" in l or "akg" in l):
                        continue
                    if key == "lemak total" and ("jenuh" in l or "trans" in l):
                        continue
                    value = None
                    for offset in range(0, 3):
                        idx = i + offset
                        if idx < len(texts):
                            target_line = texts[idx].lower()
                            if key == "lemak total" and ("jenuh" in target_line or "trans" in target_line):
                                continue
                            match = re.search(r"([0-9]+[\.,]?[0-9]*)", target_line)
                            if match:
                                value = match.group(1).replace(",", ".")
                                break 
                    if value is not None:
                        try:
                            value = float(value)
                        except:
                            pass
                        result[key] = value
    return result
