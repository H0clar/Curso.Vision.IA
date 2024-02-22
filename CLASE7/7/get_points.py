import cv2

def run(im, multi=False):
    im_disp = im.copy()
    im_draw = im.copy()
    windows_name = 'Selecciona el objeto para hacerle seguimiento'
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, im_draw)

    # Lista que contiene la parte superior izquierda y la parte inferior derecha para recortar la imagen
    pts_1 = []
    pts_2 = []
    rects = []
    run.mouse_down = False

    def callback(event, x, y, flags, param):
        nonlocal im_draw
        if event == cv2.EVENT_LBUTTONDOWN:
            run.mouse_down = True
            pts_1.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and run.mouse_down == True:
            run.mouse_down = False
            pts_2.append((x, y))
            print("objetos seleccionados en: [{},{}]".format(pts_1[-1], pts_2[-1]))

        elif event == cv2.EVENT_MOUSEMOVE and run.mouse_down == True:
            im_draw = im.copy()
            cv2.rectangle(im_draw, pts_1[-1], (x, y), (255, 255, 255), 3)
            cv2.imshow(windows_name, im_draw)

    print("Presione y suelte el mouse alrededor del objeto a seguir")

    cv2.setMouseCallback(windows_name, callback)
    print("Presione la tecla P para continuar con los puntos seleccionados")
    print("Presione la tecla D para borrar el último punto seleccionado")
    print("Presiona la tecla Q para salir")

    while True:
        windows_name_2 = "tracker de objetos"
        for pt1, pt2 in zip(pts_1, pts_2):
            rects.append([pt1[0], pt1[1], pt2[0], pt2[1]])
            cv2.rectangle(im_disp, pt1, pt2, (255, 255, 255), 3)

        # Mostrar imágenes cortadas
        cv2.imshow(windows_name_2, im_disp)
        key = cv2.waitKey(30)
        if key == ord('p'):
            # Presiona s para volver a seleccionar los puntos
            cv2.destroyAllWindows()
            point = [(tl + br) for tl, br in zip(pts_1, pts_2)]
            corrected_point = check_point(point)
            return corrected_point

        elif key == ord('q'):
            print("Eliminado sin guardar")
            exit()

        elif key == ord('d'):
            if run.mouse_down == False and pts_1:
                print("Objeto borrado en [{},{}]".format(pts_1[-1], pts_2[-1]))
                pts_1.pop()
                pts_2.pop()
                im_disp = im.copy()
            else:
                print("No hay objetos para borrar")

    cv2.destroyAllWindows()
    point = [(tl + br) for tl, br in zip(pts_1, pts_2)]
    corrected_point = check_point(point)
    return corrected_point


def check_point(points):
    out = []
    for point in points:
        if point[0] < point[2]:
            minx = point[0]
            maxx = point[2]
        else:
            minx = point[2]
            maxx = point[0]

        if point[1] < point[3]:
            miny = point[1]
            maxy = point[3]
        else:
            miny = point[3]
            maxy = point[1]

        out.append((minx, miny, maxx, maxy))

    return out

if __name__ == '__main__':
    # Añade la lógica para cargar la imagen o el video
    im = cv2.imread("ruta_de_la_imagen.jpg")
    points = run(im)
