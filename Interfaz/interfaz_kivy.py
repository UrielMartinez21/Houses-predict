import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image as PilImage
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.graphics import Color, Rectangle
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.lang import Builder
from torchvision import models
from torch import nn
import torch
import normalizar as nom
from plyer import filechooser

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Cargar el modelo preentrenado ResNet50
        self.image_features_ = models.resnet50(pretrained=True)

        # Eliminar la última capa de clasificación (fully connected) de ResNet50
        self.image_features_ = nn.Sequential(*list(self.image_features_.children())[:-1])

        # Procesamiento de las características numéricas (4 características)
        self.numeric_features_ = nn.Sequential(
            nn.Linear(4, 64),  # Aquí usas 4 datos numéricos
            nn.GELU(),  # GELU en lugar de ReLU
            nn.Dropout(),
            nn.Linear(64, 64*3),
            nn.GELU(),  # GELU
            nn.Dropout(),
            nn.Linear(64*3, 64*3*3),
            nn.GELU(),  # GELU
        )

        # Capa final que combina las características visuales y numéricas
        self.combined_features_ = nn.Sequential(
            nn.Linear(2048 + 64*3*3, 64*3*3*2*2),  # 2048 provienen de ResNet50 + numéricas
            nn.GELU(),  # GELU
            nn.Dropout(),
            nn.Linear(64*3*3*2*2, 64*3*3*2),
            nn.GELU(),  # GELU
            nn.Linear(64*3*3*2, 64),
            nn.Linear(64, 1),  # Predicción final
        )

    def forward(self, x, y):
        # Pasar las imágenes por ResNet50 para obtener las características visuales
        x = self.image_features_(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        # Pasar las características numéricas por la red densa
        y = self.numeric_features_(y)

        # Combinar características visuales y numéricas
        z = torch.cat((x, y), dim=1)
        # print(z.shape)

        # Pasar las características combinadas por las capas finales
        z = self.combined_features_(z)

        return z.squeeze(1)
    
# Ruta y tamaño deseado para la imagen
def ajustar_imagen(ruta_original, ruta_ajustada, tamano):
    with PilImage.open(ruta_original) as img:
        img = img.resize(tamano)
        img.save(ruta_ajustada)
    
# Iniciar la ventana de chat en un hilo separado, si no está ya abierta
def abrir_chat():
    global ventana_ayuda_abierta
    if not ventana_ayuda_abierta:
        ventana_ayuda_abierta = True
        threading.Thread(target=mostrar_chat).start()

# Diccionario de preguntas y respuestas
RESPUESTAS = {
    "¿Es necesario toda la información?": "Sí, es necesario para un avalúo preciso.",
    "¿Es inmediato?": "El proceso es rápido, pero puede tomar unos minutos.",
    "¿Es gratuito?": "Sí, este servicio es gratuito.",
    "¿Cómo funciona?": "Nuestro sistema utiliza inteligencia artificial para evaluar tu inmueble."
}

# Variable global para controlar la ventana de ayuda
ventana_ayuda_abierta = False

# Función para mostrar respuesta a preguntas frecuentes
def mostrar_respuesta(pregunta):
    respuesta = RESPUESTAS.get(pregunta, "No tengo una respuesta para esa pregunta.")
    messagebox.showinfo("Respuesta", respuesta)

# Función para mostrar el centro de ayuda en Tkinter
def mostrar_chat():
    global ventana_ayuda_abierta

    # Crear una ventana oculta de Tkinter para inicializar el entorno
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter

    # Crear ventana de chat
    chat = tk.Toplevel(root)
    chat.title("Centro de Ayuda")
    chat.geometry("350x500")

    # Al cerrar la ventana, actualizar la variable de control y destruir el entorno de Tkinter
    def cerrar_chat():
        global ventana_ayuda_abierta
        ventana_ayuda_abierta = False
        chat.destroy()
        root.quit()  # Cierra el entorno de Tkinter por completo

    chat.protocol("WM_DELETE_WINDOW", cerrar_chat)

    # Título
    tk.Label(chat, text="Centro de ayuda", font=("Helvetica", 16), bg="#d3d3d3").pack(fill="x")

    # Sección de preguntas comunes
    tk.Label(chat, text="Preguntas comunes", font=("Helvetica", 12)).pack(anchor="w", padx=10, pady=10)
    
    # Botones para preguntas comunes
    for pregunta in RESPUESTAS.keys():
        tk.Button(chat, text=pregunta, font=("Helvetica", 10), relief="ridge",
                  command=lambda p=pregunta: mostrar_respuesta(p)).pack(padx=10, pady=5, anchor="w")
    
    # Sección de chat
    tk.Label(chat, text="Chat", font=("Helvetica", 12)).pack(anchor="w", padx=10, pady=10)
    
    # Mensaje de chat inicial
    tk.Label(chat, text="¡Hola! Ingresa tu pregunta y verificaré si tengo una\nrespuesta a tu pregunta o a una pregunta similar.",
             font=("Helvetica", 10), relief="ridge", wraplength=320, justify="left", padx=10, pady=5).pack(anchor="w", padx=10, pady=5)

    # Botón para cerrar
    tk.Button(chat, text="Cerrar", command=cerrar_chat).pack(side="bottom", pady=10)

    # Ejecutar el bucle de eventos de Tkinter
    root.mainloop()

class ColoredBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(1, 0.6, 0, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, value):
        # Actualizar el tamaño y la posición del rectángulo de fondo
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class AvaluoApp(App):
    def build(self):
        #self.modelo = nom.obtener_modelo()
        self.title = "Avalúo de Inmuebles"
        self.root = BoxLayout(orientation='vertical')
        self.crear_interfaz_inicio()
        return self.root

    def crear_interfaz_inicio(self):
        self.root.clear_widgets()

        # Layout superior que contendrá dos secciones
        layout_superior = BoxLayout(orientation='horizontal', size_hint=(1, 0.65))

        # Sección izquierda-arriba con la imagen de la casa
        layout_1 = BoxLayout(size_hint=(0.6, 1), orientation='vertical')
        imagen_1 = Image(source='assets/imagen.jpeg', allow_stretch=True, keep_ratio=True)
        layout_1.add_widget(imagen_1)

        # Sección derecha-arriba con el texto y el botón
        layout_2 = ColoredBoxLayout(size_hint=(0.4, 1), orientation='vertical')

        texto_inicio = Label(text="[b]Bienvenido a 'Price4House',[/b]\nesta aplicación te ayuda a\notener el avalúo de tu\ninmueble mediante\ninteligencia artificial.\n\nSolicita el avalúo de tu\ninmueble de forma inmediata.",
                             font_size=24, markup=True, halign='center', valign='middle')
        texto_inicio.bind(size=texto_inicio.setter('text_size'))
        layout_2.add_widget(texto_inicio)

        btnEA = Button(text="Empezar avalúo", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btnEA.bind(on_press=self.empezar_avaluo)
        layout_2.add_widget(btnEA)

        # Layout inferior que contendrá las otras dos secciones
        layout_inferior = BoxLayout(orientation='horizontal', size_hint=(1, 0.35))

        # Sección izquierda-abajo con la descripción
        layout_3 = ColoredBoxLayout(size_hint=(0.6, 1), orientation='vertical')

        texto_desc = Label(text="[b]¿Qué es un avalúo?[/b]\nUn avalúo es un documento técnico que permite estimar el valor de un inmueble a partir de sus características físicas, de ubicación, de uso, y del análisis del mercado inmobiliario.",
                            font_size=24, markup=True, halign='center', valign='top')
        texto_desc.bind(size=texto_desc.setter('text_size'))
        layout_3.add_widget(texto_desc)

        layout_btn_ayuda = BoxLayout(orientation='horizontal', spacing=10, size_hint=(None, None), pos_hint={'center_x': 0.45})

        ajustar_imagen("Interfaz/assets/chat.png", "Interfaz/assets/chat_ajustado.png", (30, 30))
        btn_ayuda = Button(background_normal="Interfaz/assets/chat_ajustado.png", size_hint=(None, None), size=(30, 30))
        btn_ayuda.bind(on_release=lambda instance: abrir_chat())

        label_btn_ayuda = Label(text="Centro de ayuda", font_size=22, size_hint=(None, None), size=(150, 40))

        layout_btn_ayuda.add_widget(btn_ayuda)
        layout_btn_ayuda.add_widget(label_btn_ayuda)

        layout_3.add_widget(layout_btn_ayuda)

        layout_btn_manual = BoxLayout(orientation='horizontal', spacing=10, size_hint=(None, None), pos_hint={'center_x': 0.44})

        ajustar_imagen("Interfaz/assets/manual.png", "Interfaz/assets/manual_ajustado.png", (30, 30))
        btn_manual = Button(background_normal="Interfaz/assets/manual_ajustado.png", size_hint=(None, None), size=(30, 30))
        #btn_manual.bind()

        label_btn_manual = Label(text="Manual de Usuario", font_size=22, size_hint=(None, None), size=(170, 40))

        layout_btn_manual.add_widget(btn_manual)
        layout_btn_manual.add_widget(label_btn_manual)

        layout_3.add_widget(layout_btn_manual)

        # Sección derecha-abajo con la imagen
        layout_4 = ColoredBoxLayout(size_hint=(0.4, 1), orientation='vertical')
        imagen_2 = Image(source='assets/imagen.jpeg', allow_stretch=True, keep_ratio=True)
        layout_4.add_widget(imagen_2)

        # Añadir las secciones superiores al layout principal
        layout_superior.add_widget(layout_1)
        layout_superior.add_widget(layout_2)

        # Añadir las secciones inferiores al layout principal
        layout_inferior.add_widget(layout_3)
        layout_inferior.add_widget(layout_4)

        # Añadir los layouts superior e inferior al root
        self.root.add_widget(layout_superior)
        self.root.add_widget(layout_inferior)

        self.limpiar_form(0)

    def empezar_avaluo(self, *args):
        self.root.clear_widgets()
        layout = ColoredBoxLayout(orientation='vertical', padding=10, spacing=10)

        btn_regresar = Button(text="Regresar", background_color=(1, 0, 0, 1), color=(1, 1, 1, 1), size_hint=(None, None), size=(150, 40), pos_hint={'x': 0.01})
        btn_regresar.bind(on_release=self.pantalla_inicio)
        layout.add_widget(btn_regresar)

        instrucciones = Label(text="Proporciona la información solicitada\npara generar el avalúo de tu inmueble.", font_size=28, halign="center", height=70)
        layout.add_widget(instrucciones)

        fila_text_1 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        num_habitaciones_label = Label(
            text="Número de habitaciones:",
            size_hint=(0.5, 1)
        )

        num_banos_label = Label(
            text="Número de baños:",
            size_hint=(0.5, 1)
        )

        fila_text_1.add_widget(num_habitaciones_label)
        fila_text_1.add_widget(num_banos_label)
        layout.add_widget(fila_text_1)

        fila_1 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        self.num_habitaciones_in = Spinner(
            background_color=(1, 0.2, 0.5, 1),
            text=self.num_habitaciones,
            values=["1", "2", "3", "4", "5", "6", "7", "8"],
            size_hint=(0.5, 1)
        )

        if(self.num_habitaciones == ""):
            self.num_habitaciones_in.text = "1"
        else:
            self.num_habitaciones_in.text = self.num_habitaciones
        
        self.num_banos_in = Spinner(
            background_color=(1, 0.2, 0.5, 1),
            text=self.num_habitaciones,
            values=["0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5"],
            size_hint=(0.5, 1)
        )

        if(self.num_banos == ''):
            self.num_banos_in.text = "0.5"
        else:
            self.num_banos_in.text = self.num_banos

        fila_1.add_widget(self.num_habitaciones_in)
        fila_1.add_widget(self.num_banos_in)
        layout.add_widget(fila_1)

        fila_text_2 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        num_habitaciones_label = Label(
            text="Área en metros cuadrados:",
            size_hint=(0.5, 1)
        )

        num_banos_label = Label(
            text="Código postal::",
            size_hint=(0.5, 1)
        )

        fila_text_2.add_widget(num_habitaciones_label)
        fila_text_2.add_widget(num_banos_label)
        layout.add_widget(fila_text_2)

        fila_2 = BoxLayout(orientation='horizontal', size_hint=(1, 0.5), spacing=10)

        if(self.metros_cuadrados == ''):
            self.metros_cuadrados_in = TextInput(hint_text="Ingresa el área en metros cuadrados", multiline=False, halign="center", size_hint=(0.5, 1))
        else:
            self.metros_cuadrados_in = TextInput(text=self.metros_cuadrados, multiline=False, halign="center", size_hint=(0.5, 1))

        if(self.codigo_postal == ''):
            self.codigo_postal_in = TextInput(hint_text="Ingresa el código postal", multiline=False, halign="center", size_hint=(0.5, 1))
        else:
            self.codigo_postal_in = TextInput(text=self.codigo_postal, multiline=False, halign="center", size_hint=(0.5, 1))
            
        fila_2.add_widget(self.metros_cuadrados_in)
        fila_2.add_widget(self.codigo_postal_in)
        layout.add_widget(fila_2)

        # Botones para cargar imágenes
        fila_3 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_cocina = Button(text="Seleccionar imagen de la cocina", size_hint=(0.35, 1))
        btn_img_cocina.bind(on_release=lambda *args: self.obtener_img(0))
        label_img_cocina = Label(text=self.imagenes[0], halign="left", valign="middle", font_size=18, size_hint=(0.5, 1))
        label_img_cocina.bind(size=label_img_cocina.setter('text_size'))
        fila_3.add_widget(btn_img_cocina)
        fila_3.add_widget(label_img_cocina)
        layout.add_widget(fila_3)

        fila_4 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_habitacion = Button(text="Seleccionar imagen de la habitación principal", size_hint=(0.35, 1))
        btn_img_habitacion.bind(on_release=lambda *args: self.obtener_img(1))
        label_img_habitacion = Label(text=self.imagenes[1], halign="left", valign="middle", size_hint=(0.5, 1))
        label_img_habitacion.bind(size=label_img_habitacion.setter('text_size'))
        fila_4.add_widget(btn_img_habitacion)
        fila_4.add_widget(label_img_habitacion)
        layout.add_widget(fila_4)

        fila_5 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_bano = Button(text="Seleccionar imagen del baño", size_hint=(0.35, 1))
        btn_img_bano.bind(on_release=lambda *args: self.obtener_img(2))
        label_img_bano = Label(text=self.imagenes[2], halign="left", valign="middle", size_hint=(0.5, 1))
        label_img_bano.bind(size=label_img_bano.setter('text_size'))
        fila_5.add_widget(btn_img_bano)
        fila_5.add_widget(label_img_bano)
        layout.add_widget(fila_5)

        fila_6 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10, pos_hint={'x': 0.15})
        btn_img_frontal = Button(text="Seleccionar imagen frontal del inmueble", size_hint=(0.35, 1))
        btn_img_frontal.bind(on_release=lambda *args: self.obtener_img(3))
        label_img_frontal = Label(text=self.imagenes[3], halign="left", valign="middle", size_hint=(0.5, 1))
        label_img_frontal.bind(size=label_img_frontal.setter('text_size'))
        fila_6.add_widget(btn_img_frontal)
        fila_6.add_widget(label_img_frontal)
        layout.add_widget(fila_6)

        # Botones para generar avalúo y limpiar
        btn_generar = Button(text="Generar Avalúo", background_color=(0, 1, 0, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_generar.bind(on_release=self.validar_campos)
        layout.add_widget(btn_generar)

        btn_limpiar = Button(text="Limpiar", background_color=(1, 0, 0, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_limpiar.bind(on_release=lambda *args: self.limpiar_form(1))
        layout.add_widget(btn_limpiar)

        self.edicion = False

        self.root.add_widget(layout)

    def pantalla_inicio(self, *args):
        self.crear_interfaz_inicio()

    """ def obtener_img(self, tipo_imagen):
        root = tk.Tk()
        root.withdraw()

        ruta_img = filedialog.askopenfilename(title="Selecciona una imagen", 
                                                filetypes=(("Archivos de imagen", "*.png;*.jpg;*.jpeg"), ("Todos los archivos", "*.*")))

        self.num_habitaciones = self.num_habitaciones_in.text
        self.num_banos = self.num_banos_in.text
        self.metros_cuadrados = self.metros_cuadrados_in.text
        self.codigo_postal = self.codigo_postal_in.text

        if ruta_img:
            self.imagenes[tipo_imagen] = ruta_img
            print(self.imagenes)
            self.empezar_avaluo() """
    
    def obtener_img(self, tipo_imagen):
        # Abrir el cuadro de diálogo para seleccionar una imagen
        ruta_img = filechooser.open_file(
            title="Selecciona una imagen",
            filters=[("Archivos de imagen", "*.png;*.jpg;*.jpeg"), ("Todos los archivos", "*.*")]
        )

        # Validar si el usuario seleccionó una imagen
        if ruta_img:
            ruta_img = ruta_img[0]  # `filechooser.open_file` devuelve una lista de rutas

            # Actualizar las variables de entrada
            self.num_habitaciones = self.num_habitaciones_in.text
            self.num_banos = self.num_banos_in.text
            self.metros_cuadrados = self.metros_cuadrados_in.text
            self.codigo_postal = self.codigo_postal_in.text

            # Guardar la imagen en el diccionario
            self.imagenes[tipo_imagen] = ruta_img
            print(self.imagenes)

            # Continuar con el proceso
            self.empezar_avaluo()

    def resultado_avaluo(self, *args):
        self.root.clear_widgets()
        layout = ColoredBoxLayout(orientation='vertical', padding=10, spacing=10)

        btn_inicio = Button(text="Inicio", background_color=(0, 0.5, 0.5, 1), size_hint=(None, None), size=(150, 40), pos_hint={'x': 0.01})
        btn_inicio.bind(on_release=self.pantalla_inicio)
        layout.add_widget(btn_inicio)

        if(self.resultado != 0.0):
            resultado = Label(text="Avalúo generado correctamente", font_size=28, halign="center", size_hint=(0.5, None), pos_hint={'x': 0.25})
            with resultado.canvas.before:
                Color(0, 1, 0, 1)
                rect_fondo = Rectangle(size=resultado.size, pos=resultado.pos)

            resultado.bind(size=lambda instance, value: setattr(rect_fondo, 'size', instance.size),
                        pos=lambda instance, value: setattr(rect_fondo, 'pos', instance.pos))

            layout.add_widget(resultado)

            info = Label(text="El precio estimado del inmueble es de:\n $ " + str(self.resultado) + " USD", font_size=28, halign="center")
            layout.add_widget(info)
        else:
            resultado = Label(text="Error al generar el avalúo", font_size=28, halign="center", size_hint=(0.5, None), pos_hint={'x': 0.25})
            with resultado.canvas.before:
                Color(1, 0, 0, 1)
                rect_fondo = Rectangle(size=resultado.size, pos=resultado.pos)

            resultado.bind(size=lambda instance, value: setattr(rect_fondo, 'size', instance.size),
                        pos=lambda instance, value: setattr(rect_fondo, 'pos', instance.pos))

            layout.add_widget(resultado)

        linea_blanca = Label(size_hint=(1, None), size=(0, 10))
        with linea_blanca.canvas.before:
            Color(1, 1, 1, 1)
            rect_fondo_linea = Rectangle(size=linea_blanca.size, pos=linea_blanca.pos)

        linea_blanca.bind(size=lambda instance, value: setattr(rect_fondo_linea, 'size', instance.size),
                          pos=lambda instance, value: setattr(rect_fondo_linea, 'pos', instance.pos))

        layout.add_widget(linea_blanca)

        if(self.resultado != 0.0):
            instrucciones_1 = Label(text="Oprima el botón de descargar para obtener el\ndocumento con la información del avalúo", font_size=28, halign="center")
            layout.add_widget(instrucciones_1)

            btn_descargar = Button(text="Descargar", background_color=(0, 0, 1, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
            #btn_descargar.bind(on_release=self.resultado_avaluo)
            layout.add_widget(btn_descargar)

        layout.add_widget(Label(size_hint=(None, None)))

        instrucciones_2 = Label(text="Oprima el botón de editar para regresar a la pantalla\nanterior y modificar la información ingresada", font_size=28, halign="center")
        layout.add_widget(instrucciones_2)

        btn_editar = Button(text="Editar", background_color=(1, 1, 0, 1), size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_editar.bind(on_release=self.validar_campos)
        layout.add_widget(btn_editar)

        layout.add_widget(Label(size_hint=(None, None)))

        self.edicion = True

        self.root.add_widget(layout)

    def limpiar_form(self, accion):
        self.imagenes = ["Ninguna imagen subida"] * 4
        self.num_habitaciones = ''
        self.num_banos = ''
        self.metros_cuadrados = ''
        self.codigo_postal = ''
        self.resultado = ''

        if(accion == 1):
            self.empezar_avaluo() 

    def validar_campos(self, *args):
        try:
            self.num_habitaciones = self.num_habitaciones_in.text
            self.num_banos = self.num_banos_in.text
            self.metros_cuadrados = self.metros_cuadrados_in.text
            self.codigo_postal = self.codigo_postal_in.text

            if(self.edicion):
                self.empezar_avaluo()
            else:
                # Conexión con modelo y estimación
                #self.resultado = "100"
                try:
                    self.resultado = nom.normalizar_inmueble(self.imagenes[0], self.imagenes[1], self.imagenes[2], self.imagenes[3], 
                                    [self.num_habitaciones, self.num_banos, self.metros_cuadrados, self.codigo_postal])
                    self.resultado_avaluo()
                except:
                    self.resultado = 0.0
                    self.resultado_avaluo()
        except ValueError as e:
            print(f"Error de validación: {e}")

if __name__ == "__main__":
    AvaluoApp().run()