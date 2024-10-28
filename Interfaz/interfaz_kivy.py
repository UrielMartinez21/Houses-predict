import tkinter as tk
from tkinter import filedialog
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle

class ColoredBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(1, 0.5, 0, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, value):
        # Actualizar el tamaño y la posición del rectángulo de fondo
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class AvaluoApp(App):
    def build(self):
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
        imagen_1 = Image(source='assets/imagen.jpeg')
        layout_1.add_widget(imagen_1)

        # Sección derecha-arriba con el texto y el botón
        layout_2 = ColoredBoxLayout(size_hint=(0.4, 1), orientation='vertical')

        texto_inicio = Label(text="[b]Bienvenido a 'Price4House',[/b]\nesta aplicación te ayuda a\notener el avalúo de tu\ninmueble mediante\ninteligencia artificial.",
                             markup=True, halign='center', valign='middle')
        texto_inicio.bind(size=texto_inicio.setter('text_size'))
        layout_2.add_widget(texto_inicio)

        btnEA = Button(text="Empezar avalúo", size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        #btnEA.bind(on_press=self.empezar_avaluo)
        layout_2.add_widget(btnEA)

        # Layout inferior que contendrá las otras dos secciones
        layout_inferior = BoxLayout(orientation='horizontal', size_hint=(1, 0.35))

        # Sección izquierda-abajo con la descripción
        layout_3 = ColoredBoxLayout(size_hint=(0.6, 1), orientation='vertical')

        texto_desc = Label(text="[b]¿Qué es un avalúo?[/b]\n\nUn avalúo es un documento técnico que permite estimar el valor de un inmueble a partir de sus características físicas, de ubicación, de uso, y del análisis del mercado inmobiliario.",
                            markup=True, halign='left', valign='top')
        texto_desc.bind(size=texto_desc.setter('text_size'))
        layout_3.add_widget(texto_desc)

        btn_help = Button(text="Centro de ayuda", size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_manual = Button(text="Manual de usuario", size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        layout_3.add_widget(btn_help)
        layout_3.add_widget(btn_manual)

        # Sección derecha-abajo con la imagen
        layout_4 = ColoredBoxLayout(size_hint=(0.4, 1), orientation='vertical')
        imagen_2 = Image(source='assets/imagen.jpeg')
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

        self.imagenes = ["Ninguna imagen subida"] * 4

    """ def empezar_avaluo(self, *args):
        self.root.clear_widgets()
        layout = ColoredBoxLayout(orientation='vertical', padding=10, spacing=10)

        btn_regresar = Button(text="Regresar", size_hint=(None, None), size=(150, 40), pos_hint={'x': 0.01})
        btn_regresar.bind(on_release=self.pantalla_inicio)
        layout.add_widget(btn_regresar)

        instrucciones = Label(text="Proporciona la información solicitada para generar el avalúo de tu inmueble.", font_size=28, halign="center")
        layout.add_widget(instrucciones)

        fila_1 = BoxLayout(orientation='horizontal', size_hint=(1, 1), spacing=10)
        self.num_habitaciones = TextInput(hint_text="Número de habitaciones", multiline=False, halign="center", size_hint=(0.5, 1))
        self.num_banos = TextInput(hint_text="Número de baños", multiline=False, halign="center", size_hint=(0.5, 1))
        fila_1.add_widget(self.num_habitaciones)
        fila_1.add_widget(self.num_banos)
        layout.add_widget(fila_1)

        fila_2 = BoxLayout(orientation='horizontal', size_hint=(1, 1), spacing=10)
        self.metros_cuadrados = TextInput(hint_text="Área en metros cuadrados", multiline=False, halign="center", size_hint=(0.5, 1))
        self.codigo_postal = TextInput(hint_text="Código postal", multiline=False, halign="center", size_hint=(0.5, 1))
        fila_2.add_widget(self.metros_cuadrados)
        fila_2.add_widget(self.codigo_postal)
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

        # Botones para generar avalúo, limpiar y regresar
        btn_generar = Button(text="Generar Avalúo", size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        #btn_generar.bind(on_release=self.resultado_avaluo)
        btn_generar.bind(on_release=self.validar_campos)
        layout.add_widget(btn_generar)

        btn_limpiar = Button(text="Limpiar", size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        #btn_limpiar.bind(on_release=self.resultado_avaluo)
        layout.add_widget(btn_limpiar)

        self.edicion = False

        self.root.add_widget(layout) """

    """ def resultado_avaluo(self, *args):
        self.root.clear_widgets()
        layout = ColoredBoxLayout(orientation='vertical', spacing=10)

        layout.add_widget(Label(size_hint=(None, 0.1)))

        resultado = Label(text="Avalúo generado correctamente", font_size=28, halign="center", size_hint=(0.5, None), height=70, pos_hint={'x': 0.25})
        with resultado.canvas.before:
            Color(0, 1, 0, 1)
            rect_fondo = Rectangle(size=resultado.size, pos=resultado.pos)

        resultado.bind(size=lambda instance, value: setattr(rect_fondo, 'size', instance.size),
                       pos=lambda instance, value: setattr(rect_fondo, 'pos', instance.pos))

        layout.add_widget(resultado)

        info = Label(text="El precio estimado del inmueble es de:\n $ 1,249,000.00 USD", font_size=28, size_hint_y=None, height=140, halign="center")
        layout.add_widget(info)

        linea_blanca = Label(size_hint=(1, None), size=(0, 10))
        with linea_blanca.canvas.before:
            Color(0, 0, 0, 1)
            rect_fondo_linea = Rectangle(size=linea_blanca.size, pos=linea_blanca.pos)

        linea_blanca.bind(size=lambda instance, value: setattr(rect_fondo_linea, 'size', instance.size),
                          pos=lambda instance, value: setattr(rect_fondo_linea, 'pos', instance.pos))

        layout.add_widget(linea_blanca)

        layout.add_widget(Label(size_hint=(None, None))) 

        instrucciones_1 = Label(text="Oprima el botón de descargar para obtener el\ndocumento con la información del avalúo", font_size=28, size_hint_y=None, height=70, halign="center")
        layout.add_widget(instrucciones_1)

        btn_descargar = Button(text="Descargar", size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        #btn_descargar.bind(on_release=self.resultado_avaluo)
        layout.add_widget(btn_descargar)

        layout.add_widget(Label(size_hint=(None, 0.1)))

        instrucciones_2 = Label(text="Oprima el botón de editar para regresar a la pantalla\nanterior y modificar la información ingresada", font_size=28, size_hint_y=None, height=70, halign="center")
        layout.add_widget(instrucciones_2)

        btn_editar = Button(text="Editar", size_hint=(None, None), size=(150, 40), pos_hint={'center_x': 0.5})
        btn_editar.bind(on_release=self.validar_campos)
        layout.add_widget(btn_editar)

        self.edicion = True

        layout.add_widget(Label(size_hint=(None, None))) 

        self.root.add_widget(layout) """       

    def pantalla_inicio(self, *args):
        self.crear_interfaz_inicio()

    """ def validar_campos(self, *args):
        try:
            print(int(self.num_habitaciones.text), int(self.num_banos.text))

            if(self.edicion):
                self.empezar_avaluo()
            else:
                self.resultado_avaluo()
        except ValueError as e:
            print(f"Error de validación: {e}") """

    """ def obtener_img(self, tipo_imagen):
        root = tk.Tk()
        root.withdraw()

        ruta_img = filedialog.askopenfilename(title="Selecciona una imagen", 
                                                filetypes=(("Archivos de imagen", "*.png;*.jpg;*.jpeg"), ("Todos los archivos", "*.*")))

        if ruta_img:
            self.imagenes[tipo_imagen] = ruta_img
            print(self.imagenes)
            self.empezar_avaluo() """

if __name__ == "__main__":
    AvaluoApp().run()