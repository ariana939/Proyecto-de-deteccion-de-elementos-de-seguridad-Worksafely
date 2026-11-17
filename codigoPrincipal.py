import cv2
import numpy as np
import onnxruntime as ort
from gpiozero import LED, OutputDevice

led_verde = LED(17)
led_rojo = LED(27)
rele = OutputDevice(22, active_high=False, initial_value=False)

# variable de estado para enclavamiento del relé
soldadora_habilitada = False

# estado inicial seguro al arrancar el programa
rele.off()
led_verde.off()
led_rojo.on()

# Modelo de IA (onnx)
ruta_del_modelo = "/home/pipi/Desktop/proyectoSoldadura/weights.onnx"
clases = ["lentes de proteccion", "lentes personales", "persona", "prenda protectora"]
confianza_minima = 0.25


# :(
opciones = ort.SessionOptions()
opciones.intra_op_num_threads = 3
session = ort.InferenceSession(ruta_del_modelo, sess_options=opciones, providers=["CPUExecutionProvider"]) # carga archivo weights.onnx
input_name = session.get_inputs()[0].name
_, _, input_h, input_w = session.get_inputs()[0].shape # le pregunta que forma tiene las fotos que espera recibir

# Cámara y control en vivo/tiempo real
cap = cv2.VideoCapture(0) # abre cámara y las 2 líneas de abajo es para fija la cap en pixeles
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # para que la cam no acumule foto viejas y solo se quede con una

print("Sistema iniciado. Teclas para el video: 'r' = reiniciar bloqueo y 'q' = salir.")

try:
    while True: # análisis, foto tras foto
        for _ in range(3):# descarta 3 fotos viejas acumuladas
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:# si se desconecta la cám o si ni estaba conectada
            print("Error: no se recibió imagen de la cámara.")
            break


        # preparación de la foto para la IA :(
        img = cv2.resize(frame, (input_w, input_h))# achica o agranda la imagen según lo que necesite el modelo
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0 
        img = np.expand_dims(img, axis=0)


        # deducciones y lectura de resultados
        outputs = session.run(None, {input_name: img})[0] # le pasa la imagen preparada al modelo onnx y obtirne los resultados/predicciones
        predicciones = np.squeeze(outputs).T # :((

        visto = {} # :(
        for fila in predicciones:
            puntajes = fila[4:]
            clase_id = int(np.argmax(puntajes))
            conf = float(puntajes[clase_id])
            if conf >= confianza_minima and clase_id < len(clases):
                nombre = clases[clase_id]
                if conf > visto.get(nombre, 0):
                    visto[nombre] = round(conf, 2)

        print("Detectado:", visto)

        # verifica qué elementos obligatorios faltan
        faltantes = []
        if "lentes de proteccion" not in visto:
            faltantes.append("lentes de proteccion")
        if "prenda protectora" not in visto:
            faltantes.append("prenda protectora")

        cumple_todo = len(faltantes) == 0 # si está vacia

        # Lógica de enclavamiento y advertancia
        if not soldadora_habilitada:
            # Etapa 1: Espera habilitación inicial
            if cumple_todo:
                soldadora_habilitada = True
                rele.on()
                led_verde.on()
                led_rojo.off()
                linea1 = "HABILITADO"
                linea2 = f"Veo: {', '.join(visto.keys())}"
                color = (0, 255, 0)
            else:
                rele.off()
                led_verde.off()
                led_rojo.on()
                linea1 = "BLOQUEADO" # bloqueo inicial
                linea2 = f"Falta: {', '.join(faltantes)}"# :((
                color = (0, 0, 255)
        else:
            # Etapa 2: Ya habilitada (el relé no se corta, si falta un elemento solo avisa con LED rojo)
            rele.on()
            if cumple_todo:
                led_verde.on()
                led_rojo.off()
                linea1 = "HABILITADO" # ya habilitado, osea sigue cumpliendo todo
                linea2 = f"Veo: {', '.join(visto.keys())}"
                color = (0, 255, 0)
            else:
                led_verde.off()
                led_rojo.on()
                linea1 = "ALERTA" # pero el relé sigue activo
                linea2 = f"Falta: {', '.join(faltantes)}"
                color = (0, 165, 255) 
        
        # pantalla
        cv2.putText(frame, linea1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2) # para que dibuje la línea 1 en la pantalla
        cv2.putText(frame, linea2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1) # para que dibuje la línea 2 en la pantalla
        cv2.imshow("Seguridad Soldadura", frame) # abre y actualiza ventana de la camara


        # teclado
        tecla = cv2.waitKey(1) & 0xFF # espera 1 milisegundo para refrescar ventana y se fija/captura si toco una tecla
        if tecla == ord("q"): # sale del while
            break
        elif tecla == ord("r"): # reinicia el sistema para volver a bloquear el relé si hace falta
            soldadora_habilitada = False
            rele.off()
            led_verde.off()
            led_rojo.on()
            print("Sistema reiniciado a estado BLOQUEADO.")


# se ejecuta al salir del while, apaga todo, fuerza al relé tambien, deja de usar la cámara
finally:
    rele.off()
    led_verde.off()
    led_rojo.off()
    cap.release()
    cv2.destroyAllWindows() # cierra ventana de la camara

