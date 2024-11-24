from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, white, black
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import os
import winreg
import shutil

def obtener_carpeta_descargas():
    try:
        # Abrir la clave de registro
        clave = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
        # Obtener el valor de la clave de Descargas
        carpeta_descargas, _ = winreg.QueryValueEx(clave, "{374DE290-123F-4565-9164-39C4925E467B}")
        winreg.CloseKey(clave)
        return carpeta_descargas
    except Exception as e:
        print(f"Error al obtener la carpeta de Descargas: {e}")
        return os.path.join(os.path.expanduser("~"), "Downloads")
    
def colocar_manual_descargas(ruta):
    ruta = os.path.dirname(ruta)
    carpeta_descargas = obtener_carpeta_descargas()
    ruta_manual = ruta + "/Docs/Manual_de_usuario.pdf"

    shutil.copy(ruta_manual, carpeta_descargas + "/Manual_de_usuario.pdf")

def generar_reporte_avaluo(kitchen_path, bedroom_path, bathroom_path, frontal_path, numeric_features, resultado):
    """ bathroom_path = 'C:/Users/adria/Documents/Python/TT2/Houses-predict/Dataset/Houses-dataset/301_bathroom.jpg'
    kitchen_path = 'C:/Users/adria/Documents/Python/TT2/Houses-predict/Dataset/Houses-dataset/301_kitchen.jpg'
    frontal_path = 'C:/Users/adria/Documents/Python/TT2/Houses-predict/Dataset/Houses-dataset/301_frontal.jpg'
    bedroom_path = 'C:/Users/adria/Documents/Python/TT2/Houses-predict/Dataset/Houses-dataset/301_bedroom.jpg'

    numeric_features = [5,5.0,4014,92880]

    resultado = 100000.00 """

    c = canvas.Canvas(obtener_carpeta_descargas() + "/Reporte_Avalúo.pdf")

    # Dimensiones de la página
    ancho, alto = letter

    # Dibujar el encabezado (rectángulo azul claro)
    color_azul_claro = Color(0.67, 0.84, 0.9)  # Azul claro (RGB normalizado entre 0 y 1)
    c.setFillColor(color_azul_claro)
    c.rect(-1, alto - 50, ancho, 50, fill=1)

    # Agregar texto del encabezado (centrado, letras blancas)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(ancho / 2, alto - 35, "Reporte de Avalúo")

    # Coordenadas iniciales y tamaño de imágenes
    margen_x = 50
    margen_y = 100
    ancho_imagen = 200
    alto_imagen = 100
    espacio_x = (ancho - 2 * margen_x - 2 * ancho_imagen) / 1  # Espacio entre columnas
    espacio_y = 20  # Espacio entre filas

    # Función para agregar una imagen con su leyenda
    def agregar_imagen(canvas, path, leyenda, x, y):
        # Agregar la leyenda
        canvas.setFont("Helvetica", 12)
        canvas.drawCentredString(x + ancho_imagen / 2, y + alto_imagen + 10, leyenda)
        # Agregar la imagen
        canvas.drawImage(path, x, y, width=ancho_imagen, height=alto_imagen)

    # Coordenadas de las imágenes
    imagenes = [
        (bathroom_path, "Imagen del Baño"),
        (kitchen_path, "Imagen de la Cocina"),
        (frontal_path, "Imagen Frontal del Inmueble"),
        (bedroom_path, "Imagen de la Habitación Principal")
    ]

    # Agregar las imágenes en un diseño de 2x2
    x_inicio = margen_x
    y_inicio = alto - 200  # Debajo del encabezado
    for i, (path, leyenda) in enumerate(imagenes):
        fila = i // 2
        columna = i % 2
        x = x_inicio + columna * (ancho_imagen + espacio_x)
        y = y_inicio - fila * (alto_imagen + espacio_y + 20)
        agregar_imagen(c, path, leyenda, x, y)

    # Dibujar el recuadro azul claro para los datos
    alto_recuadro = 150  # Altura del recuadro ajustada para la tabla
    y_recuadro = y_inicio - 2 * (alto_imagen + espacio_y) - 90  # Justo debajo de las imágenes
    c.setFillColor(color_azul_claro)
    c.rect(-1, y_recuadro, ancho, alto_recuadro, fill=1)

    # Crear datos para la tabla
    tabla_datos = [
        ["Dato", "Descripción"],
        ["Número de Habitaciones", numeric_features[0]],
        ["Número de Baños", numeric_features[1]],
        ["Área en Metros Cuadrados", numeric_features[2]],
        ["Código Postal", numeric_features[3]],
    ]

    # Crear la tabla
    tabla = Table(tabla_datos, colWidths=[200, 200])
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),  # Fondo de la cabecera
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Texto blanco en la cabecera
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),         # Texto centrado
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Fuente negrita en cabecera
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),      # Fuente normal en datos
        ('FONTSIZE', (0, 0), (-1, -1), 10),              # Tamaño de fuente
        ('GRID', (0, 0), (-1, -1), 1, colors.black),     # Líneas de la tabla
        ('BACKGROUND', (0, 1), (-1, -1), colors.orange), # Fondo claro en datos
    ]))

    # Coordenadas para dibujar la tabla dentro del recuadro azul
    x_tabla = 106  # Margen izquierdo
    y_tabla = y_recuadro + alto_recuadro - 100  # Posición inicial de la tabla

    # Dibujar la tabla
    tabla.wrapOn(c, ancho, alto)
    tabla.drawOn(c, x_tabla, y_tabla)

    # Agregar información adicional (texto centrado) debajo de la tabla, dentro del recuadro azul
    texto_precio = f"El precio estimado del inmueble es de: $ {resultado}"
    texto_ancho = c.stringWidth(texto_precio, "Helvetica", 14)

    # Posicionar el texto justo debajo de la tabla dentro del recuadro
    y_texto = y_tabla - 40  # Ajusta según sea necesario para que quede correctamente alineado
    c.setFont("Helvetica", 14)
    c.setFillColor(black)
    c.drawString((ancho - texto_ancho) / 2, y_texto, texto_precio)

    # Guardar el archivo
    c.save()