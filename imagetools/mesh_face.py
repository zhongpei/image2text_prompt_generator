import mediapipe as mp
import cv2
import os
import click
import shutil

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def get_image_files(image_path):
    image_files = []
    for root, dirs, files in os.walk(image_path, topdown=False):
        for name in files:
            if name.lower().endswith(".jpg") or name.lower().endswith(".png"):
                image_files.append(os.path.join(root, name))
    return image_files


@click.command()
@click.option('--image_path', '-i', show_default=True, default='./download_images', help='Path to the image')
@click.option('--output_path', '-o', show_default=True, default='./out_images', help='Path to the output ')
@click.option('--max_num_faces', show_default=True, default=1, help='Max number of faces')
@click.option('--thickness', show_default=True, default=10, help='mask drawing thickness')
@click.option('--circle_radius', show_default=True, default=1, help='mask drawing circle radius')
@click.option('--mesh_type', '-t', show_default=True, default='face', help='Mesh type: face , hand')
@click.option('--clear_output', '-c', is_flag=True, show_default=True, default=False, help='Clear output')
def mesh(image_path, output_path, max_num_faces, mesh_type='face', thickness=10, circle_radius=1, clear_output=False):
    if clear_output and os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if mesh_type.lower().find('face') != -1:
        mesh_face(image_path, output_path, max_num_faces, thickness, circle_radius)
    elif mesh_type.lower().find('hand') != -1:
        mesh_hand(image_path=image_path, output_path=output_path, thickness=thickness, circle_radius=circle_radius)


def mesh_face(image_path, output_path, max_num_faces, thickness=10, circle_radius=1):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    image_files = get_image_files(image_path)
    drawing_spec = mp_drawing.DrawingSpec(thickness=thickness, circle_radius=circle_radius)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=int(max_num_faces),
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
    ) as face_mesh:
        for idx, file in enumerate(image_files):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                print("{} : No face detected".format(file))
                continue
            annotated_image = image.copy()

            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)
                # face
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                #
                # # eyes mouth
                # mp_drawing.draw_landmarks(
                #     image=annotated_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_contours_style())

                # face eyes mouth
                # mp_drawing.draw_landmarks(
                #      image=annotated_image,
                #      landmark_list=face_landmarks,
                #      connections=mp_face_mesh.FACEMESH_IRISES,
                #      landmark_drawing_spec=None,
                #      connection_drawing_spec=mp_drawing_styles
                #      .get_default_face_mesh_iris_connections_style())
            output_file = os.path.join(output_path, str(idx) + '.png')
            print(output_file)
            cv2.imwrite(output_file, annotated_image)


def mesh_hand(image_path, output_path, thickness=10, circle_radius=1):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    # For static images:
    IMAGE_FILES = get_image_files(image_path)
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
            output_file = os.path.join(output_path, str(idx) + '.png')
            print("output_file:", output_file)
            cv2.imwrite(output_file, cv2.flip(annotated_image, 1))
            # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue
            # for hand_world_landmarks in results.multi_hand_world_landmarks:
            #     mp_drawing.plot_landmarks(
            #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)


if __name__ in {"__main__", "__mp_main__"}:
    mesh()
