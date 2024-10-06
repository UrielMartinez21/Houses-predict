from tkinter import *
from tkinter import Tk, messagebox, filedialog
from PIL import Image, ImageTk

#Pantalla 1
def pantalla_inicio():
    limpiar_frame(frame1)
    limpiar_frame(frame2)
    limpiar_frame(frame3)
    limpiar_frame(frame4)

    frame1.place(relx=0, rely=0, relwidth=0.65, relheight=0.6)
    frame2.place(relx=0.65, rely=0, relwidth=0.35, relheight=0.6)
    frame3.place(relx=0, rely=0.6, relwidth=0.65, relheight=0.4)
    frame4.place(relx=0.65, rely=0.6, relwidth=0.35, relheight=0.4)

    ventana.after(100, lambda: agregar_imagen(frame1, "imagenes/imagen.jpeg"))

    ventana.after(100, lambda: agregar_texto(frame2, """\n\nBienvenido a 'Price4House', esta aplicación te ayuda a obtener el avalúo de tu inmueble
    mediante inteligencia artificial.\n\n
    Solicita el avalúo de tu inmueble de forma inmediata.\n"""))
    ventana.after(100, lambda: agregar_boton(frame2, "Empezar avalúo", 1))

    ventana.after(100, lambda: agregar_texto(frame3, """\n¿Qué es un avalúo?\n
    Un avalúo es un documento técnico que permite estimar el valor de un inmueble a partir de sus características físicas,
    de ubicación, de uso, y de análisis del mercado inmobiliario."""))
    ventana.after(100, lambda: agregar_boton(frame3, "Centro de Ayuda", 2))
    ventana.after(100, lambda: agregar_boton(frame3, "Manual de Usuario", 3))

    ventana.after(100, lambda: agregar_imagen(frame4, "imagenes/imagen.jpeg"))

#Pantalla 2
def empezar_avaluo():
    esconder_frames()
    limpiar_frame(frame5)

    frame5.place(relx=0, rely=0, relwidth=1, relheight=1)

    Button(frame5, text="Regresar", command=inicio).place(relx=0.03, rely=0.03, relwidth=0.1)

    Label(frame5, text="""\n\nProporciona la información solicitada para generar el avalúo
    de tu inmueble.\n\n""", font=("Helvetica", 16), bg="orange", justify='center').pack()

    Label(frame5, text="Número de habitaciones:", font=("Helvetica", 12), bg="orange", justify="center").place(relx=0.25, rely=0.2, relwidth=0.21)
    entry_num_habitaciones = Entry(frame5, justify=CENTER).place(relx=0.25, rely=0.24, relwidth=0.21, relheight=0.05)

    Label(frame5, text="Número de baños:", font=("Helvetica", 12), bg="orange", justify="center").place(relx=0.55, rely=0.2, relwidth=0.21)
    entry_num_banos = Entry(frame5, justify=CENTER).place(relx=0.55, rely=0.24, relwidth=0.21, relheight=0.05)

    Label(frame5, text="Área en metros cuadrados:", font=("Helvetica", 12), bg="orange", justify="center").place(relx=0.25, rely=0.32, relwidth=0.21)
    entry_metros_cuadrados = Entry(frame5, justify=CENTER).place(relx=0.25, rely=0.36, relwidth=0.21, relheight=0.05)

    Label(frame5, text="Código postal:", font=("Helvetica", 12), bg="orange", justify="center").place(relx=0.55, rely=0.32, relwidth=0.21)
    entry_codigo_postal = Entry(frame5, justify=CENTER).place(relx=0.55, rely=0.36, relwidth=0.21, relheight=0.05)

    Label(frame5, text="Ingresa la imagen de la cocina:", font=("Helvetica", 12), bg="blue", anchor="nw").place(relx=0.15, rely=0.45, relwidth=0.35)
    button_img_cocina = Button(frame5, text="Seleccionar archivo", command=obtener_img).place(relx=0.5, rely=0.45, relwidth=0.35)

    Label(frame5, text="Ingresa la imagen de la habitación principal:", font=("Helvetica", 12), bg="blue", anchor="nw").place(relx=0.15, rely=0.52, relwidth=0.35)
    button_img_cocina = Button(frame5, text="Seleccionar archivo", command=obtener_img).place(relx=0.5, rely=0.52, relwidth=0.35)

    Label(frame5, text="Ingresa la imagen de la habitación principal:", font=("Helvetica", 12), bg="blue", anchor="nw").place(relx=0.15, rely=0.59, relwidth=0.35)
    button_img_cocina = Button(frame5, text="Seleccionar archivo", command=obtener_img).place(relx=0.5, rely=0.59, relwidth=0.35)

    Label(frame5, text="Ingresa la imagen de la habitación principal:", font=("Helvetica", 12), bg="blue", anchor="nw").place(relx=0.15, rely=0.66, relwidth=0.35)
    button_img_cocina = Button(frame5, text="Seleccionar archivo", command=obtener_img).place(relx=0.5, rely=0.66, relwidth=0.35)

    Button(frame5, text="Generar Avalúo", bg="green", command=resultado_avaluo).place(relx=0.4, rely=0.74, relwidth=0.2)
    Button(frame5, text="Limpiar", bg="red").place(relx=0.4, rely=0.79, relwidth=0.2)

#Pantalla 3
def resultado_avaluo():
    esconder_frames()
    limpiar_frame(frame6)

    frame6.place(relx=0, rely=0, relwidth=1, relheight=1)

    Button(frame6, text="Inicio", command=inicio).place(relx=0.03, rely=0.03, relwidth=0.1)

    Label(frame6, text="""Avalúo generado correctamente""", font=("Helvetica", 16), bg="green", justify='center').place(relx=0.3, rely=0.1, relwidth=0.4)

    Label(frame6, text="""El precio estimado del inmueble es de:
    $ 1,249,000.00 USD""", font=("Helvetica", 16), bg="orange", justify="center").place(relx=0.28, rely=0.2, relwidth=0.44)

    Label(frame6, bg="white", justify="center").place(rely=0.32, relwidth=1, relheight=0.01)

    Label(frame6, text="""Oprima el botón de descargar para obtener el
    documento con la información del avalúo""", font=("Helvetica", 16), bg="orange").place(relx=0.28, rely=0.36, relwidth=0.44)
    Button(frame6, text="Descargar", bg="blue").place(relx=0.45, rely=0.46, relwidth=0.1)    

    Label(frame6, text="""Oprima el botón de editar para regresar a la pantalla
    anterior y modificar la información ingresada""", font=("Helvetica", 16), bg="orange").place(relx=0.25, rely=0.54, relwidth=0.5)
    Button(frame6, text="Editar", bg="blue", command=editar).place(relx=0.45, rely=0.64, relwidth=0.1)

def esconder_frames():
    frame1.place_forget()
    frame2.place_forget()
    frame3.place_forget()
    frame4.place_forget()
    frame5.place_forget()
    frame6.place_forget()

def limpiar_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def inicio():
    esconder_frames()
    pantalla_inicio()

def editar():
    esconder_frames
    empezar_avaluo()

def obtener_img():
    img = filedialog.askopenfile()
    return img

def mostrar_respuesta(pregunta):
    respuesta = {
        "¿Es necesario toda la información?": "Sí, es necesario para un avalúo preciso.",
        "¿Es inmediato?": "El proceso es rápido, pero puede tomar unos minutos.",
        "¿Es gratuito?": "Sí, este servicio es gratuito.",
        "¿Cómo funciona?": "Nuestro sistema utiliza inteligencia artificial para evaluar tu inmueble."
    }.get(pregunta, "No tengo una respuesta para esa pregunta.")

    messagebox.showinfo("Respuesta", respuesta)

#Pantalla 4
def mostrar_chat():
    chat = Toplevel(ventana)
    chat.title("Centro de Ayuda")
    chat.geometry("350x500")
    chat.transient(ventana)
    chat.grab_set()

    # Título
    Label(chat, text="Centro de ayuda", font=("Helvetica", 16), bg="#d3d3d3").pack(fill=X)

    # Sección de preguntas comunes
    Label(chat, text="Preguntas comunes", font=("Helvetica", 12)).pack(anchor=W, padx=10, pady=10)

    # Botones para preguntas comunes
    preguntas = ["¿Es necesario toda la información?", "¿Es inmediato?", "¿Es gratuito?", "¿Cómo funciona?"]

    for pregunta in preguntas:
        Button(chat, text=pregunta, font=("Helvetica", 10), relief=RIDGE, command=lambda p=pregunta: mostrar_respuesta(p)).pack(padx=10, pady=5, anchor=W)

    # Sección de chat
    Label(chat, text="Chat", font=("Helvetica", 12)).pack(anchor=W, padx=10, pady=10)
    
    # Mensaje de chat inicial
    Label(chat, text="¡Hola!. Ingresa tu pregunta y verificaré si tengo una\nrespuesta a tu pregunta o a una pregunta similar.",
          font=("Helvetica", 10), relief=RIDGE, wraplength=320, justify=LEFT, padx=10, pady=5).pack(anchor=W, padx=10, pady=5)

    # Botón de cerrar
    Button(chat, text="Cerrar", command=chat.destroy).pack(side=BOTTOM, pady=10)

    ventana.wait_window(chat)

def agregar_imagen(frame, path):
    imagen = Image.open(path)
    imagen = imagen.resize((int(frame.winfo_width()), int(frame.winfo_height())), Image.LANCZOS)
    imagen = ImageTk.PhotoImage(imagen)
    label_imagen = Label(frame, image=imagen)
    label_imagen.image = imagen
    label_imagen.place(relwidth=1, relheight=1)

def agregar_texto(frame, texto):
    label_texto = Label(frame, text=texto, font=("Helvetica", 16), bg="orange", wraplength=frame.winfo_width(), justify='center')
    label_texto.pack()

def agregar_boton(frame, texto, funcion):
    if funcion == 1:
        button_frameEA = Button(frame, text=texto, command=empezar_avaluo)
        button_frameEA.pack()
    elif funcion == 2:
        img_boton = Image.open("imagenes/chat.png")
        img_boton = img_boton.resize((30, 30), Image.LANCZOS)
        img_boton = ImageTk.PhotoImage(img_boton)
        button_frameCA = Button(frame, text=texto, image=img_boton, bg='orange', compound=LEFT, command=mostrar_chat)
        button_frameCA.image = img_boton
        button_frameCA.pack(pady=10)
    else:
        img_boton = Image.open("imagenes/manual.png")
        img_boton = img_boton.resize((30, 30), Image.LANCZOS)
        img_boton = ImageTk.PhotoImage(img_boton)
        button_frameCA = Button(frame, text=texto, image=img_boton, bg='orange', compound=LEFT, command=empezar_avaluo)
        button_frameCA.image = img_boton
        button_frameCA.pack()

ventana = Tk()
ventana.geometry("1000x700")
ventana.title("Avalúo de Inmuebles")
ventana.iconbitmap("imagenes/icono.ico")

#Pantalla 1
frame1 = Frame(ventana)

frame2 = Frame(ventana, bg="orange")

frame3 = Frame(ventana, bg="orange")

frame4 = Frame(ventana)

#ventana.update_idletasks()

pantalla_inicio()

#Pantalla 2
frame5 = Frame(ventana, bg="orange")

#Pantalla 3
frame6 = Frame(ventana, bg="orange")

ventana.mainloop()