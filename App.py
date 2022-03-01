import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import e caricamento del set di dati 
# Il caricamento del set di dati restituisce quattro array NumPy:
# -> gli array train_images e train_labels sono il set di addestramento
# -> gli array test_images e test_labels sono il set di test del modello
# Le immagini sono array NumPy 28x28, con valori di pixel compresi tra 0 e 255. 
# Le etichette sono un array di numeri interi, compresi tra 0 e 9, che corrispondono alla classe di abbigliamento rappresentata dall'immagine
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# La label non includono i nomi delle classi, perciò li archiviamo in un'array class_names
class_names = ['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Eplorazione dei dati: formato e labels dei sue set di dati
# print(train_images.shape) #numerosità del set di addestramento e formato (60000, 28*28)
# print(len(train_labels)) #numerosità delle label di addestramento (60000)
# print(train_labels) #contenuto delle label di addestramento (numeri interi da 0 a 9)
# print(test_images.shape) #numerosità del set di test e formato (10000, 28*28)
# print(len(test_labels)) #numerosità delle label di test (60000)
# print(test_labels) #contenuto delle label di test (numeri interi da 0 a 9)

# Pre-elaborazione del set di dati
# I valori dei pixel rientrano nell'intervallo da 0 a 255 (plot della prima immagine del set di addestramento).
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# Questi valori vanno riportati in un intervallo da 0 a 1 prima di inviarli al modello di rete neurale.  
# È importante che il set di addestramento e il set di test siano preelaborati allo stesso modo:
train_images = train_images / 255.0
test_images = test_images / 255.0

# Verifica del formato delle immagini stampando le prime 25 immagini
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Costruzione del modello: livelli
# Il primo livello (tf.keras.layers.Flatten), trasforma il formato delle immagini da una matrice bidimensionale (di 28 x 28 pixel) 
# a una matrice unidimensionale (di 28 * 28 = 784 pixel). Questo livello non ha parametri da apprendere; riformatta solo i dati.
# Dopo che i pixel sono stati appiattiti, la rete è costituita da una sequenza di due livelli tf.keras.layers.Dense . 
# Questi sono strati neurali densamente collegati o completamente connessi. 
# Il primo strato Dense ha 128 nodi (o neuroni). 
# Il secondo (e ultimo) livello restituisce un array logit con lunghezza 10. 
# Ogni nodo contiene un punteggio che indica che l'immagine corrente appartiene a una delle 10 classi.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

# Compilazione del modello (ottimizzatore, funzione di perdita, metriche)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Allenamento del modello
model.fit(train_images, train_labels, epochs=10) #invio i dati di addestramento al modello

# Verifico l'accuratezza del modello con i dati di test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Utilizzo del modello addestrato per fare previsioni
# Associamo un livello Softmax per convertire gli output del modello in probabilità
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(predictions[0]) #print della prima previsione
# Avendo 10 classi, ogni previsione è un array di 10 elementi in cui ciascuno rappresenta
# la probabilità che l'immagine appartenga a una delle 10 classi. 
np.argmax(predictions[0])

# Esaminamo 15 previsioni e facciamo un plot per verificare tali previsioni
def plot_image(i, prediction_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                   100*np.max(prediction_array),
                                   class_names[true_label]),
                                   color=color)

def plot_value_array(i, prediction_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()