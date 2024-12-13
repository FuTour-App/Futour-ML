import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.utils as ku
import pickle

class GenreModel:
    def __init__(self, training_data, model_path='./genre_model.h5', tokenizer_path='./tokenizer.pickle', label_encoder_path='./label_encoder.pickle', max_sequence_len_path='max_sequence_len.txt'):
        self.training_data = training_data
        self.tokenizer = Tokenizer(num_words=30000)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.max_sequence_len_path = max_sequence_len_path
        self.num_classes = 0
        self.max_sequence_len = 0
        self.total_words = 0
        self.model = None
        self.label_encoder = LabelEncoder()

        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path) and os.path.exists(max_sequence_len_path):
            self.load_model(model_path, tokenizer_path, label_encoder_path, max_sequence_len_path)
        else:
            print("Saved model or tokenizer not found. Training a new model.")
            self.train_new_model()

    def preprocess_data(self):
        self.training_data['genre_label'] = self.label_encoder.fit_transform(self.training_data['genre'])
        self.num_classes = len(self.label_encoder.classes_)

        corpus = self.training_data['description'].tolist()
        self.tokenizer.fit_on_texts(corpus)
        self.total_words = len(self.tokenizer.word_index) + 1

        input_sequences = self.tokenizer.texts_to_sequences(corpus)
        self.max_sequence_len = max(len(seq) for seq in input_sequences)
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre'))

        X = input_sequences
        y = ku.to_categorical(self.training_data['genre_label'], num_classes=self.num_classes)
        return X, y

    def build_model(self):
        inputs = Input(shape=(self.max_sequence_len,))
        x = Embedding(self.total_words, 100, input_length=self.max_sequence_len)(inputs)
        x = Bidirectional(LSTM(150, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(GRU(64))(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_new_model(self, epochs=50, batch_size=64):
        X, y = self.preprocess_data()

        class_weights = compute_class_weight('balanced', classes=np.unique(self.training_data['genre_label']), y=self.training_data['genre_label'])
        class_weights = dict(enumerate(class_weights))

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

        self.build_model()
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        self.save_model(self.model_path, self.tokenizer_path, self.label_encoder_path, self.max_sequence_len_path)

    def save_model(self, model_path, tokenizer_path, label_encoder_path, max_sequence_len_path):
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_encoder_path, 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(max_sequence_len_path, 'w') as f:
            f.write(str(self.max_sequence_len))

    def load_model(self, model_path, tokenizer_path, label_encoder_path, max_sequence_len_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        with open(label_encoder_path, 'rb') as handle:
            self.label_encoder = pickle.load(handle)
        with open(max_sequence_len_path, 'r') as f:
            self.max_sequence_len = int(f.read())

    # def predict_genre(self, user_description):
    #     user_description = user_description.lower()
    #     token_list = self.tokenizer.texts_to_sequences([user_description])
    #     if not token_list or all(len(seq) == 0 for seq in token_list):
    #         return "Unknown Genre"
        
    #     token_list = pad_sequences(token_list, maxlen=self.max_sequence_len, padding='pre')
    #     predicted_probs = self.model.predict(token_list, verbose=0)
    #     predicted_index = np.argmax(predicted_probs)
    #     return self.label_encoder.inverse_transform([predicted_index])[0]

    def predict_next_words(self, input_text, num_words=5):
        """
        Predict the next word(s) based on input text.

        Args:
            input_text (str): Initial text to complete.
            num_words (int): Number of words to predict.

        Returns:
            str: The completed sequence.
        """
        for _ in range(num_words):
            token_list = self.tokenizer.texts_to_sequences([input_text.lower()])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len, padding='pre')  # Use self.max_sequence_len
            predicted_probs = self.model.predict(token_list, verbose=0)
            predicted_index = np.argmax(predicted_probs)
            output_word = self.tokenizer.index_word.get(predicted_index, '')
            if output_word:  # Check if the predicted index corresponds to a word
                input_text += ' ' + output_word
            else:
                break  # Stop if no valid word is predicted
        return input_text.strip()  # Return the completed sequence without leading/trailing spaces