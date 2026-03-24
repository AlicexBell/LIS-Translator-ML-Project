import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import argparse
import os
import json
import matplotlib.pyplot as plt

def train_cnn_model(data_dir, model_output_path, epochs=15):
    """
    Addestra un modello CNN (basato su MobileNetV2) per il riconoscimento dei segni LIS.

    Args:
        data_dir (str): Percorso della cartella contenente le immagini,
                        organizzate in sottocartelle per classe.
        model_output_path (str): Percorso dove salvare il modello addestrato (.h5).
        epochs (int): Numero di epoche per l'addestramento.
    """
    
    # --- 1. Caricamento Dati ---
    print(f"INFO: Caricamento dati da: {data_dir}")
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"La cartella {data_dir} non esiste.")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"INFO: Trovate {num_classes} classi: {class_names}")

    # Ottimizzazione della pipeline di dati
    print("INFO: Ottimizzazione della pipeline di caricamento dati...")
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 2. Data Augmentation ---
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # --- 3. Creazione Modello (Transfer Learning) ---
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    print("INFO: Architettura del modello creata.")
    model.summary()

    # --- 4. Compilazione e Addestramento ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"INFO: Inizio addestramento per {epochs} epoche...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs
    )
    print("INFO: Addestramento completato.")

    # --- 5. Salvataggio Modello ---
    # Assicurati che l'estensione sia .keras invece di .h5
    if not model_output_path.endswith('.keras'):
        model_output_path = model_output_path.replace('.h5', '.keras')
        if not model_output_path.endswith('.keras'):
            model_output_path += '.keras'

    model.save(model_output_path)
    print(f"✅ Modello salvato con successo in: {model_output_path}")

    # Salva anche i nomi delle classi per uso futuro nell'applicazione
    class_names_path = os.path.join(os.path.dirname(model_output_path), 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"INFO: Nomi delle classi salvati in: {class_names_path}")

    # --- 6. (Opzionale) Salvataggio Grafico Performance ---
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plot_path = os.path.join(output_dir, 'training_performance.png')

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.savefig(plot_path)
        print(f"INFO: Grafico delle performance salvato in: {plot_path}")
    except Exception as e:
        print(f"WARN: Impossibile salvare il grafico delle performance: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script per addestrare una CNN per il riconoscimento LIS.")
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=True,
        help="Percorso alla cartella contenente le immagini del dataset."
    )
    parser.add_argument(
        '--output-path', 
        type=str, 
        required=True,
        help="Percorso completo (incluso il nome del file .h5) dove salvare il modello addestrato."
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=15,
        help="Numero di epoche per l'addestramento (default: 15)."
    )
    args = parser.parse_args()

    train_cnn_model(args.data_dir, args.output_path, args.epochs)
