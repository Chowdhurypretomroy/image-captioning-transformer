"""
Image Captioning with CNN + Transformer

Architecture:
    - CNN encoder: EfficientNetB0 (pretrained on ImageNet, frozen) extracts
      visual features from the input image.
    - Transformer encoder: refines the visual features.
    - Transformer decoder: generates a caption autoregressively, attending
      to the encoded visual features.

Dataset: Flickr8k (images + 5 captions per image).
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization



# Reproducibility


SEED = 111
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Configuration


# Path to the extracted Flickr8k images
IMAGES_PATH = "Flicker8k_Dataset"

# Image dimensions expected by EfficientNetB0
IMAGE_SIZE = (299, 299)

# Vocabulary cap and caption length cap
VOCAB_SIZE = 10000
SEQ_LENGTH = 25

# Embedding and feed-forward dimensions
EMBED_DIM = 512
FF_DIM = 512

# Training
BATCH_SIZE = 64
EPOCHS = 30        # was 2 in the original; 30 is a realistic minimum
AUTOTUNE = tf.data.AUTOTUNE



# Data extraction 


def extract_flickr8k():
    """Extract the Flickr8k zip files if they haven't been extracted yet.

    Uses the standard library so this works outside Colab/Jupyter too.
    """
    import zipfile

    if not os.path.isdir(IMAGES_PATH):
        if os.path.exists("Flickr8k_Dataset.zip"):
            with zipfile.ZipFile("Flickr8k_Dataset.zip", "r") as z:
                z.extractall()
        else:
            raise FileNotFoundError(
                "Flickr8k_Dataset.zip not found. Download it first."
            )

    if not os.path.exists("Flickr8k.token.txt"):
        if os.path.exists("Flickr8k_text.zip"):
            with zipfile.ZipFile("Flickr8k_text.zip", "r") as z:
                z.extractall()
        else:
            raise FileNotFoundError(
                "Flickr8k_text.zip not found. Download it first."
            )



# Caption loading


def load_captions_data(filename):
    """Load captions and map them to their images

    Returns:
        caption_mapping: {image_path: [caption1, ..., caption5]}
        text_data:       flat list of all captions, each wrapped in
                         <start> ... <end> tokens
    """
    with open(filename) as f:
        caption_data = f.readlines()

    caption_mapping = {}
    text_data = []
    images_to_skip = set()

    for line in caption_data:
        line = line.rstrip("\n")
        # Image and caption are tab-separated; image name has a "#N" suffix
        img_name, caption = line.split("\t")
        img_name = img_name.split("#")[0]
        img_name = os.path.join(IMAGES_PATH, img_name.strip())

        # Filter out captions that are too short or too long
        tokens = caption.strip().split()
        if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
            images_to_skip.add(img_name)
            continue

        if img_name.endswith("jpg") and img_name not in images_to_skip:
            caption = "<start> " + caption.strip() + " <end>"
            text_data.append(caption)
            caption_mapping.setdefault(img_name, []).append(caption)

    
    for img_name in images_to_skip:
        caption_mapping.pop(img_name, None)

    return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the {image: captions} dict into train and validation dicts."""
    all_images = list(caption_data.keys())
    if shuffle:
        np.random.shuffle(all_images)

    n_train = int(len(all_images) * train_size)
    training_data = {k: caption_data[k] for k in all_images[:n_train]}
    validation_data = {k: caption_data[k] for k in all_images[n_train:]}
    return training_data, validation_data



# Text vectorization


# Strip everything except <, > (which we need for <start>/<end> tokens)
strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
strip_chars = strip_chars.replace("<", "").replace(">", "")


def custom_standardization(input_string):
    """Lowercase and strip punctuation while preserving < and >."""
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")



# Image pipeline
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)


def decode_and_resize(img_path):
    """Read a JPEG, resize to IMAGE_SIZE, and convert to float32 in [0, 1]."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


# Positional embedding

class PositionalEmbedding(layers.Layer):
    """Token + sinusoidal-style learned positional embedding

    Combines token embeddings (scaled by sqrt(embed_dim) as is standard for
    transformers) with learned positional embeddings
    """

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs) * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # Pad tokens are 0; mask them out for downstream attention layers.
        return tf.math.not_equal(inputs, 0)



# CNN feature extractor
def get_cnn_model():
    """EfficientNetB0 truncated before the classification head, frozen."""
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    # Flatten the spatial dims 
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    return keras.models.Model(base_model.input, base_model_out)



# Transformer encoder block

class TransformerEncoderBlock(layers.Layer):
    """Standard pre-norm transformer encoder block: LN -> MHA -> residual."""

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_proj = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training=False, mask=None):
        # Project visual features to the model dimension
        x = self.dense_proj(inputs)
        x = self.layernorm_1(x)

        attention_output = self.attention(
            query=x, value=x, key=x,
            attention_mask=None,
            training=training,
        )
        # Residual + norm
        return self.layernorm_2(x + attention_output)



# Transformer decoder block

class TransformerDecoderBlock(layers.Layer):
    """Decoder block: self-attention (causal) + cross-attention to encoder + FFN."""

    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training=False, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        # Build padding/combined masks; default to causal-only when no
        # padding mask is supplied (fixes the UnboundLocalError in the original)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            padding_mask = None
            combined_mask = causal_mask

        # Self-attention with combined causal + padding mask
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Cross-attention to the encoder outputs
        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        # Position-wise FFN
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)

        return self.out(ffn_out)

    def get_causal_attention_mask(self, inputs):
        """Lower-triangular mask so position t can only attend to positions <= t."""
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)



# Full captioning model

class ImageCaptioningModel(keras.Model):
    """Glues CNN -> encoder -> decoder and defines train/test steps."""

    def __init__(self, cnn_model, encoder, decoder,
                 num_captions_per_image=5, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]   # decoder input (everything but last token)
        batch_seq_true = batch_seq[:, 1:]   # target (everything but first token)
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0.0
        batch_acc = 0.0

        if self.image_aug is not None:
            batch_img = self.image_aug(batch_img)

        # Compute image features once per batch
        img_embed = self.cnn_model(batch_img)
        with tf.GradientTape() as tape:
            for i in range(self.num_captions_per_image):
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )
                batch_loss += loss
                batch_acc += acc
            avg_loss = batch_loss / float(self.num_captions_per_image)

        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(avg_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0.0
        batch_acc = 0.0

        img_embed = self.cnn_model(batch_img)
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]



# Learning rate schedule

class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup, then constant."""

    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.post_warmup_learning_rate * (global_step / warmup_steps)
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_lr,
            lambda: self.post_warmup_learning_rate,
        )



# Inference (greedy decoding)

def generate_caption(model, vectorization, valid_images, index_lookup,
                     max_decoded_sentence_length):
    """Pick a random validation image, display it, and print the predicted caption."""
    sample_img_path = np.random.choice(valid_images)
    sample_img = decode_and_resize(sample_img_path)

    # Display: image is in [0, 1] float32, so just imshow directly.
    plt.imshow(sample_img.numpy())
    plt.axis("off")
    plt.show()

    # CNN features
    img = tf.expand_dims(sample_img, 0)
    img = model.cnn_model(img)
    encoded_img = model.encoder(img, training=False)

    # Greedy decode token by token
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":   # was " <end>" in the original — never matched
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    print("Predicted caption:", decoded_caption)


# Main

def main():
    
    extract_flickr8k()

    # 2. Load and split captions
    captions_mapping, text_data = load_captions_data("Flickr8k.token.txt")
    train_data, valid_data = train_val_split(captions_mapping)
    print(f"Number of training samples:   {len(train_data)}")
    print(f"Number of validation samples: {len(valid_data)}")

    # 3. Build and adapt the text vectorizer
    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)

    # 4. Build the tf.data pipeline
    def process_input(img_path, captions):
        return decode_and_resize(img_path), vectorization(captions)

    def make_dataset(images, captions):
        ds = tf.data.Dataset.from_tensor_slices((images, captions))
        ds = ds.shuffle(BATCH_SIZE * 8)
        ds = ds.map(process_input, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
    valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

    # 5. Build the model
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model,
        encoder=encoder,
        decoder=decoder,
        image_aug=image_augmentation,
    )

    # 6. Compile
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True
    )
    num_train_steps = len(train_dataset) * EPOCHS
    num_warmup_steps = num_train_steps // 15
    lr_schedule = LRSchedule(
        post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps
    )
    caption_model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=cross_entropy,
    )

    # 7. Train
    caption_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        callbacks=[early_stopping],
    )

    # 8. Inference on a few validation images
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1
    valid_images = list(valid_data.keys())

    for _ in range(3):
        generate_caption(
            caption_model, vectorization, valid_images,
            index_lookup, max_decoded_sentence_length
        )


if __name__ == "__main__":
    main()
