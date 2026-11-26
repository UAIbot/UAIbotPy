import numpy as np
from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.cylinder import *
from uaibot.simobjects.ball import *

from .links import *


def _create_ur_ur3e(
    htm: np.ndarray = np.identity(4),
    name: str = "",
    color: str = "#009fe3",
    opacity: float = 1.0,
):
    """
    Cria a estrutura interna de um robô Universal Robots UR3e para o UaiBot.
    """
    # --- Validação dos Parâmetros de Entrada ---
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception(
            "O parâmetro 'htm' deve ser uma matriz de transformação homogênea 4x4."
        )

    if not Utils.is_a_name(name):
        raise Exception(
            "O parâmetro 'name' deve ser uma string. Apenas caracteres 'a-z', 'A-Z', '0-9' e '_' são permitidos e não deve começar com um número."
        )

    if not Utils.is_a_color(color):
        raise Exception(
            "O parâmetro 'color' deve ser uma cor compatível com HTML (ex: '#ff6600')."
        )

    if (not Utils.is_a_number(opacity)) or not 0 <= opacity <= 1:
        raise Exception("O parâmetro 'opacity' deve ser um float entre 0 e 1.")

    # --- Parâmetros DH (seguindo o estilo do KR5) ---
    # [theta, d, a, alpha, tipo_junta]
    link_info = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "theta" rotação em z
        [0.15185, 0.0, 0.0, 0.13105, 0.08535 + 0.02, 0.0921],  # "d" translação em z
        [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0],  # "alpha" rotação em x
        [
            0.0,
            -0.24355,
            -0.21320,
            0.0,
            0.0,
            0.0,
        ],  # "a" translação em x (valor corrigido para seguir a convenção correta)
        [0, 0, 0, 0, 0, 0],  # tipo da junta (0 para rotacional)
    ]
    # O 'a' do DH para os links 2 e 3 são negativos.

    n = 6

    # --- Modelo de Colisão ---
    col_model = [[] for _ in range(n)]

    htm_obj = [[None for _ in range(8)] for _ in range(6)]
    htm_obj[0][0] = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -0.0, 1.0, -0.0469],
            [0.0, -1.0, -0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[1][0] = np.matrix(
        [
            [-0.0, -1.0, -0.0, 0.2454],
            [1.0, -0.0, -0.0, 0.0],
            [-0.0, -0.0, 1.0, 0.118],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[1][1] = np.matrix(
        [
            [-0.0, 0.0, -1.0, 0.1254],
            [1.0, 0.0, -0.0, -0.0],
            [-0.0, -1.0, -0.0, 0.12],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[1][2] = np.matrix(
        [
            [-0.0, -1.0, -0.0, 0.0004],
            [1.0, -0.0, -0.0, -0.0],
            [-0.0, -0.0, 1.0, 0.118],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[2][0] = np.matrix(
        [
            [-0.0, 0.0, -1.0, 0.1886],
            [1.0, 0.0, -0.0, -0.0],
            [-0.0, -1.0, -0.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[2][1] = np.matrix(
        [
            [-0.0, 0.0, -1.0, 0.1086],
            [1.0, 0.0, -0.0, -0.0],
            [-0.0, -1.0, -0.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[2][2] = np.matrix(
        [
            [-0.0, -1.0, -0.0, -0.0014],
            [1.0, -0.0, -0.0, -0.0],
            [-0.0, -0.0, 1.0, 0.045],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[3][0] = np.matrix(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -0.0, 1.0, 0.0039],
            [0.0, 1.0, 0.0, 0.0014],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[3][1] = np.matrix(
        [
            [-1.0, -0.0, 0.0, 0.0],
            [0.0, -1.0, -0.0, -0.0011],
            [0.0, 0.0, 1.0, 0.0414],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[4][0] = np.matrix(
        [
            [0.0, 1.0, -0.0, 0.0011],
            [-0.0, -0.0, -1.0, 0.034],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[4][1] = np.matrix(
        [
            [0.0, 1.0, 0.0, 0.0011],
            [1.0, -0.0, -0.0, 0.004],
            [-0.0, 0.0, -1.0, -0.0025],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[5][0] = np.matrix(
        [
            [0.0, 1.0, 0.0, 0.0011],
            [1.0, -0.0, -0.0, 0.004],
            [-0.0, 0.0, -1.0, -0.0231],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[5][1] = np.matrix(
        [
            [0.0, 1.0, -0.0, 0.0011],
            [-0.0, -0.0, -1.0, -0.021],
            [-1.0, 0.0, 0.0, -0.0201],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[5][2] = np.matrix(
        [
            [0.0, 1.0, -0.0, 0.0011],
            [-0.0, -0.0, -1.0, 0.004],
            [-1.0, 0.0, 0.0, 0.0279],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[5][3] = np.matrix(
        [
            [0.0, 1.0, -0.0, 0.0011],
            [-0.0, -0.0, -1.0, -0.006],
            [-1.0, 0.0, 0.0, 0.1079],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # Exact ±45° components keep these collision frames on SO(3).
    c = np.sqrt(0.5)
    htm_obj[5][4] = np.matrix(
        [
            [c, c, 0.0, -0.0389],
            [0.0, 0.0, -1.0, -0.001],
            [-c, c, 0.0, 0.1529],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[5][5] = np.matrix(
        [
            [-c, c, 0.0, 0.0411],
            [0.0, 0.0, -1.0, -0.001],
            [-c, -c, 0.0, 0.1529],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[5][6] = np.matrix(
        [
            [0.0, 1.0, 0.0, 0.0511],
            [1.0, -0.0, -0.0, -0.001],
            [-0.0, 0.0, -1.0, 0.1979],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm_obj[5][7] = np.matrix(
        [
            [0.0, 1.0, 0.0, -0.0489],
            [1.0, -0.0, -0.0, -0.001],
            [-0.0, 0.0, -1.0, 0.1979],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    col_model[0].append(
        Cylinder(
            htm=htm_obj[0][0],
            name=name + "_C0",
            radius=0.067,
            height=0.21,
            color="red",
            opacity=0.3,
        )
    )
    col_model[1].append(
        Cylinder(
            htm=htm_obj[1][0],
            name=name + "_C11",
            radius=0.052,
            height=0.13,
            color="green",
            opacity=0.3,
        )
    )
    col_model[1].append(
        Cylinder(
            htm=htm_obj[1][1],
            name=name + "_C12",
            radius=0.05,
            height=0.2,
            color="blue",
            opacity=0.3,
        )
    )
    col_model[1].append(
        Cylinder(
            htm=htm_obj[1][2],
            name=name + "_C13",
            radius=0.05,
            height=0.12,
            color="red",
            opacity=0.3,
        )
    )
    col_model[2].append(
        Ball(
            htm=htm_obj[2][0],
            name=name + "_C21",
            radius=0.05,
            color="green",
            opacity=0.3,
        )
    )
    col_model[2].append(
        Cylinder(
            htm=htm_obj[2][1],
            name=name + "_C22",
            radius=0.04,
            height=0.2,
            color="blue",
            opacity=0.3,
        )
    )
    col_model[2].append(
        Cylinder(
            htm=htm_obj[2][2],
            name=name + "_C23",
            radius=0.035,
            height=0.09,
            color="red",
            opacity=0.3,
        )
    )
    col_model[3].append(
        Cylinder(
            htm=htm_obj[3][0],
            name=name + "_C31",
            radius=0.035,
            height=0.09,
            color="green",
            opacity=0.3,
        )
    )
    col_model[3].append(
        Cylinder(
            htm=htm_obj[3][1],
            name=name + "_C32",
            radius=0.035,
            height=0.045,
            color="blue",
            opacity=0.3,
        )
    )
    col_model[4].append(
        Cylinder(
            htm=htm_obj[4][0],
            name=name + "_C41",
            radius=0.035,
            height=0.025,
            color="red",
            opacity=0.3,
        )
    )
    col_model[4].append(
        Cylinder(
            htm=htm_obj[4][1],
            name=name + "_C42",
            radius=0.038,
            height=0.098,
            color="green",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Cylinder(
            htm=htm_obj[5][0],
            name=name + "_C51",
            radius=0.038,
            height=0.046,
            color="blue",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Cylinder(
            htm=htm_obj[5][1],
            name=name + "_C52",
            radius=0.01,
            height=0.028,
            color="red",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Ball(
            htm=htm_obj[5][2],
            name=name + "_C53",
            radius=0.05,
            color="green",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Box(
            htm=htm_obj[5][3],
            name=name + "_C54",
            width=0.09,
            depth=0.07,
            height=0.06,
            color="blue",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Box(
            htm=htm_obj[5][4],
            name=name + "_C55",
            width=0.075,
            depth=0.04,
            height=0.035,
            color="red",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Box(
            htm=htm_obj[5][5],
            name=name + "_C56",
            width=0.075,
            depth=0.04,
            height=0.035,
            color="green",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Cylinder(
            htm=htm_obj[5][6],
            name=name + "_C57",
            radius=0.021,
            height=0.04,
            color="red",
            opacity=0.3,
        )
    )
    col_model[5].append(
        Cylinder(
            htm=htm_obj[5][7],
            name=name + "_C58",
            radius=0.021,
            height=0.04,
            color="blue",
            opacity=0.3,
        )
    )

    # --- Modelos 3D e suas Transformações ---

    # <<< ALTERAÇÃO AQUI: Matriz de transformação homogênea 4x4 explícita para a rotação de 90 graus em X

    htm_L0 = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, -0.0, 0.0],
            [0.0, -0.0, 1.0, 0.1169],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    htm_L1 = np.matrix(
        [[1, 0, 0, 0], [0, 0, 1, -0.055 + 0.02], [0, 1, 0, 0], [0, 0, 0, 1]]
    )

    htm_L2 = np.matrix(
        [[0, -1, 0, 0.1485 - 0.02], [1, 0, 0, 0], [0, 0, -1, 0.144], [0, 0, 0, 1]]
    )

    htm_L3 = np.matrix(
        [
            [0, -1, 0, 0.0628 - 0.02],
            [1, 0, 0, 0],  # ------------
            [0, 0, -1, 0.035],
            [0, 0, 0, 1],
        ]
    )

    # --------------------------------------

    htm_L4 = np.matrix(
        [[-1, 0, 0, 0.00], [0, -1, 0, 0.0185], [0, 0, 1, -0.015 + 0.02], [0, 0, 0, 1]]
    )

    htm_L5 = np.matrix(
        [
            [1, 0, 0, 0.0],
            [0, -1, 0, 0.0105 - 0 * 0.02],
            [0, 0, -1, -0.0176],
            [0, 0, 0, 1],
        ]
    )
    htm_L6 = np.matrix(
        [[1, 0, 0, 0.0], [0, 1, 0, 0.0 - 0 * 0.02], [0, 0, 1, -0.022], [0, 0, 0, 1]]
    )

    htm_L7 = np.matrix(
        [[1, 0, 0, 0.0], [0, 0, 1, 0.0 - 0 * 0.02], [0, -1, 0, 0.012], [0, 0, 0, 1]]
    )

    # A base do UR3e é o seu primeiro link, então 'base_3d_obj' fica vazio.
    base_3d_obj = [
        Model3D(
            "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/UniversalUR3/UR3e_Link1A.obj",
            0.001,
            htm_L0,
            MeshMaterial(
                metalness=0.7, clearcoat=1, roughness=0.5, color=color, opacity=opacity
            ),
        )
    ]

    # Lista de objetos 3D para cada link
    link_3d_obj = []

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/UniversalUR3/UR3e_Link1B.obj",
                0.001,
                htm_L1,
                MeshMaterial(
                    metalness=0.7,
                    clearcoat=1,
                    roughness=0.5,
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/UniversalUR3/UR3e_Link2.obj",
                0.001,
                htm_L2,
                MeshMaterial(
                    metalness=0.7,
                    clearcoat=1,
                    roughness=0.5,
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/UniversalUR3/UR3e_Link3.obj",
                0.001,
                htm_L3,
                MeshMaterial(
                    metalness=0.7,
                    clearcoat=1,
                    roughness=0.5,
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/UniversalUR3/UR3e_Link4.obj",
                0.001,
                htm_L4,
                MeshMaterial(
                    metalness=0.7,
                    clearcoat=1,
                    roughness=0.5,
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/UniversalUR3/UR3e_Link5.obj",
                0.001,
                htm_L5,
                MeshMaterial(
                    metalness=0.7,
                    clearcoat=1,
                    roughness=0.5,
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/UniversalUR3/UR3e_Link6.obj",
                0.001,
                htm_L6,
                MeshMaterial(
                    metalness=0.7,
                    clearcoat=1,
                    roughness=0.5,
                    color="#555555",
                    opacity=opacity,
                ),
            ),
            Model3D(
                "https://raw.githubusercontent.com/gfmgg0/Uaibot_UR/refs/heads/main/model_gripper.obj",
                0.001,
                htm_L7,
                MeshMaterial(
                    metalness=0.7,
                    clearcoat=1,
                    roughness=0.5,
                    color="#555555",
                    opacity=opacity,
                ),
            ),
        ]
    )

    # --- Criação dos Links ---
    links = []
    for i in range(n):
        # Acessa os parâmetros DH da lista 'link_info' de forma organizada
        links.append(
            Link(
                i,
                link_info[0][i],
                link_info[1][i],
                link_info[2][i],
                link_info[3][i],
                link_info[4][i],
                link_3d_obj[i],
            )
        )

        # Anexa os objetos de colisão a cada link
        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # --- Configurações Finais ---
    # q0 = [0, -np.pi / 2, 0.0, -np.pi / 2, -np.pi / 2, 0.0]
    q0 = [0, -np.pi / 2, 0.0, -np.pi / 2, -np.pi / 2, 0.0]

    # Limites das juntas em radianos
    # Limites de junta (graus → radianos)
    joint_limits = (np.pi / 180) * np.matrix(
        [
            [-360, 360],  # Base
            [-360, 360],  # Ombro
            [-360, 360],  # Cotovelo
            [-360, 360],  # Pulso 1
            [-360, 360],  # Pulso 2
            [-np.inf, np.inf],  # Pulso 3 (rotação contínua)
        ]
    )

    # htm_base_0 e htm_n_eef são as transformações da base para o primeiro frame DH e do último para o end-effector.
    htm_base_0 = np.matrix([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]])
    htm_n_eef = np.matrix(
        [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.2], [0, 0, 0, 1]]
    )

    return base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits
