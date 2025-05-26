from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def crear_pdf(nombre_archivo):
    c = canvas.Canvas(nombre_archivo, pagesize=letter)
    width, height = letter

    # --- Encabezado superior personalizado ---
    header_height = 120
    # Fondo azul claro
    c.setFillColor(colors.HexColor('#cfe2f3'))
    c.rect(30, height-header_height-30, width-60, header_height, fill=1, stroke=1)

    # Título principal
    c.setFillColor(colors.HexColor('#003366'))
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height-45, "REPORTE DE ANALISIS DE GASES DISUELTOS EN ACEITE DE ORIGEN MINERAL")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(width-100, height-35, "2-R-232/2")

    # Logo ENDE (espacio reservado)
    c.setFillColor(colors.white)
    c.rect(40, height-header_height-20, 120, 60, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#003366'))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(55, height-header_height+10, "ENDE")
    c.setFont("Helvetica", 10)
    c.drawString(55, height-header_height-5, "TRANSMISIÓN")

    # Logo LAB (espacio reservado)
    c.setFillColor(colors.white)
    c.circle(width-80, height-header_height+10, 40, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#003366'))
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(width-80, height-header_height+10, "LABORATORIO DE ACEITES")
    c.setFont("Helvetica", 7)
    c.drawCentredString(width-80, height-header_height-5, "LOGO LAB")

    # Caja No. de reporte
    c.setFillColor(colors.white)
    c.rect(width-250, height-header_height+30, 180, 30, fill=1, stroke=1)
    c.setFillColor(colors.HexColor('#003366'))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(width-240, height-header_height+45, "No.:")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(width-200, height-header_height+45, "GOM - SNC - 014 / 2024")

    # Subtítulo laboratorio
    c.setFillColor(colors.HexColor('#003366'))
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, height-header_height+10, "LABORATORIO DE ACEITE DIELÉCTRICO")

    # Datos cliente y fecha
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(60, height-header_height-10, "CLIENTE:")
    c.setFont("Helvetica", 9)
    c.drawString(120, height-header_height-10, "CIASA")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(60, height-header_height-30, "FECHA:")
    c.setFont("Helvetica", 9)
    c.drawString(120, height-header_height-30, "23 de febrero de 2024")

    # Cajas para datos
    c.setStrokeColor(colors.black)
    c.rect(115, height-header_height-15, 180, 15, fill=0, stroke=1)
    c.rect(115, height-header_height-35, 180, 15, fill=0, stroke=1)

    c.save()

if __name__ == "__main__":
    crear_pdf("reporte_encabezado.pdf")