from flask import Flask, render_template, Response
from flask import request as rq
from werkzeug.utils import secure_filename
import numpy as np
import os, sys, glob
import cv2
import yolov5
import base64
import imutils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ejecutando modelo Yolo
def detected_joint_yolo(tipo_img):
    valor = ''
    img = ''

    try:
        # Extrayendo path de las imágenes en la carpeta
        for path in glob.glob('uploads/*'):
            # Extrayendo la ruta de la imagen cargada al servidor
            img = path
            break
            # Nombre del archivo (imagen)
            #nomb = os.path.split(img)
            #nomb = os.path.splitext(nomb[1])
            #print(nomb[0])


        # Seleccionado el modelo Yolo a implementar
        if tipo_img == '2':
            modelo = 'YOLOv5_XRAY'
        elif tipo_img == '1':
            modelo = 'YOLOv5_LASER'
        else:
            modelo = 'YOLOv5_RGB'

        # Cargando los pesos entrenados al modelo
        model = yolov5.load('PESOS_YOLO/'+modelo+'.pt')
          
        # Parametros para el modelo
        ''' === Umbral de confianza

        Solo identificará a los objetos que tengan un probabilidad 
        de 50% o más de ser uno de los objetos con el cual fue
        entrenado'''
        model.conf = 0.50


        ''' === Umbral de superposición 
        La superposición (intersección sobre unión) se permite entre 
        las detecciones antes de considerarlas redundantes y eliminarlas.
        Si el valor es alto, se requerirá una mayor superposición para eliminar una detección'''
        model.iou = 0.45

        '''
        Este parámetro controla si el proceso de eliminación de 
        detecciones redundantes se realiza considerando o no las 
        clases de los objetos. Si está configurado en False, el
        algoritmo tendrá en cuenta las clases y mantendrá solo una 
        detección por clase para cada objeto detectado. 

        Si está configurado en True, se eliminarán las detecciones 
        redundantes sin importar las clases.'''
        model.agnostic = False  # NMS class-agnostic

        '''
        Este parámetro controla si se permite asignar múltiples 
        etiquetas (clases) a una misma caja. 

        Si está configurado en False, cada caja solo puede tener 
        una etiqueta/clase asociada. Si está configurado en True, 
        una caja puede estar asociada con varias etiquetas/clases.
        '''
        model.multi_label = False  # NMS multiple labels per box

        '''
        Este parámetro establece cuántas detecciones se permiten 
        como máximo después de aplicar el proceso de eliminación 
        de detecciones redundantes. 

        Si el número de detecciones 
        es mayor a este valor, solo se mantendrán las detecciones
        con las mayores confianzas.
        '''
        model.max_det = 1000  # maximum number of detections per image

        # cargando imagen al modelo, considerando los argumentos indicados anteriormente
        results = model(img, augment=True)

        # Ejecutando la predicción de YOLO
        predictions = results.pred[0]

        # === RESULTADOS DE YOLO
        # Coordenadas de cada objeto identificado
        boxes = predictions[:, :4] # x1, y1, x2, y2
        boxes = list(np.array(boxes, dtype = 'int'))

        # Probabilidades de los objetos identificados
        scores = predictions[:, 4]

        # Clase a la que pertenece el objeto identificado
        categories = predictions[:, 5].tolist()
        categories = list(np.array(categories, dtype = 'int'))
        categories2 = categories.copy()

        print(modelo)
        #print(tipo_img)
        # Clases del modelo RGB
        if tipo_img == '0':
            categories = ['CMC' if c == 0 else c for c in categories]
            categories = ['MCP' if c == 1 else c for c in categories]
            categories = ['DIP' if c == 2 else c for c in categories]
            categories = ['PIP' if c == 3 else c for c in categories]

        # Clases del modelo LASER
        if tipo_img == '1':
            categories = ['MCP' if c == 0 else c for c in categories]
            categories = ['PIP' if c == 1 else c for c in categories]
            categories = ['DIP' if c == 2 else c for c in categories]

        # Clases del modelo XRAY
        if tipo_img == '2':
            categories = ['MCP' if c == 0 else c for c in categories]
            categories = ['PIP' if c == 1 else c for c in categories]
            categories = ['DIP' if c == 2 else c for c in categories]
            categories = ['CMC' if c == 3 else c for c in categories]

        # Organizando en una lista cada objeto:
        # [n][0] - CLASE
        # [n][1] - PROBABILIDAD 
        # [n][2] - COORDENADAS

        #print(categories)
        # Leyendo imagen
        img_original = cv2.imread(img)
        img2 = img_original.copy()
        colors = [(255,0,0), (0,255,0), (0, 215, 255), (0, 0, 255)]

        for elemnt in range(0, len(categories2)):
            cv2.rectangle(img2, (boxes[elemnt][0], boxes[elemnt][1]),(boxes[elemnt][2], boxes[elemnt][3]),colors[int(categories2[elemnt])], 2)
            #cv2.imshow('box_objet', img2)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #print(categories[elemnt])
        
        #print('\n\n')

        # Contabilizando los objetos identificados
        results = dict(zip(categories2,map(lambda x: categories2.count(x),categories2)))
        #print(results)

        informe = ''

        # Clases del modelo RGB
        if tipo_img == '0':
            for r in results:
                if r == 0:
                    informe += '\n - Articulaciones CMC: '+str(results[r])
                elif r == 1:
                    informe += '\n - Articulaciones MCP: '+str(results[r])
                elif r == 2:
                    informe += '\n - Articulaciones DIP: '+str(results[r])
                elif r == 3:
                    informe += '\n - Articulaciones PIP: '+str(results[r])
            
        # Clases del modelo LASER
        if tipo_img == '1':
            for r in results:
                if r == 0:
                    informe += '\n - Articulaciones MCP: '+str(results[r])
                elif r == 1:
                    informe += '\n - Articulaciones PIP: '+str(results[r])
                elif r == 2:
                    informe += '\n - Articulaciones DIP: '+str(results[r])

        # Clases del modelo XRAY
        if tipo_img == '2':
            for r in results:
                if r == 0:
                    informe += '\n - Articulaciones MCP: '+str(results[r])
                elif r == 1:
                    informe += '\n - Articulaciones PIP: '+str(results[r])
                elif r == 2:
                    informe += '\n - Articulaciones DIP: '+str(results[r])
                elif r == 3:
                    informe += '\n - Articulaciones CMC: '+str(results[r])


        valor = 'Articulaciones identificadas: '+str(len(categories))+' --- '+informe
        img_processed = img2.copy()

        (_, encodedImage2) = cv2.imencode(".jpg", img_processed)
        img_processed = bytearray(encodedImage2)

    except Exception as err:
        img_processed = np.zeros((600, 800, 3), np.uint8)
        cv2.putText(img_processed, 'Problema ocurrido:', (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_processed, str(err), (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        (_, encodedImage2) = cv2.imencode(".jpg", img_processed)
        img_processed = bytearray(encodedImage2)

        return valor, img_processed, img
    except:
        img_processed = np.zeros((600, 800, 3), np.uint8)
        cv2.putText(img_processed, 'No puede ser obtenida la imagen',
                    (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        (_, encodedImage2) = cv2.imencode(".jpg", img_processed)
        img_processed = bytearray(encodedImage2)

        return valor, img_processed, img

    return valor, img_processed, img

#Pagina raíz
@app.route("/")
def main():
    return render_template('index.html')

# Botón que inicia el procesamiento de la imagen cargada
@app.route("/procesar_img", methods=['POST'])   
def procesar_img():
    if rq.method == 'POST':
        if 'imagen' not in rq.files:
            return 'No se ha seleccionado ninguna imagen'
        
        archivo = rq.files['imagen']

        # Verificar si el archivo tiene un nombre
        if archivo.filename == '':
            return render_template('index2.html', cadena='No se ha seleccionado ninguna imagen')

        # Verificar si la extensión del archivo es válida (puedes agregar más extensiones según tus necesidades)
        if not archivo.filename.endswith(('.jpg', '.png', '.jpeg', '.gif')):
            return render_template('index2.html', cadena='Formato de imagen no válido')

        # Guardar el archivo en el directorio de carga
        filename = secure_filename(archivo.filename)
        archivo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Aquí puedes hacer cualquier otro procesamiento con la imagen si es necesario
        dato2 = rq.form.get('tipo_img')

        # Ejecutar YOLO
        datos, imagen_base64, path_img = detected_joint_yolo(dato2)

        # Eliminando la imagen temporal del servidor
        if path_img != '':
            os.remove(path_img)

        # Codificando imagen a 64 bits
        imagen_base64 = base64.b64encode(imagen_base64).decode('utf-8')

        return render_template('index3.html', imagen_base64=imagen_base64, cadena=datos)
    
if __name__ == "__main__":
    app.run(debug=False, port=5100)