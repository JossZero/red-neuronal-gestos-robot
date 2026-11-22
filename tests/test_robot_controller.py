"""
Pruebas automáticas para el módulo de control del robot y filtro de seguridad.
"""

from src.robot_controller import FiltroSeguridad, SimuladorRobot


def test_filtro_seguridad_baja_confianza_fuerza_stop():
    """Verifica que una predicción con confianza menor al umbral no active movimiento."""
    filtro = FiltroSeguridad(umbral_confianza=0.75, fotogramas_confirmacion=3)

    # 3 predicciones seguidas de FORWARD pero con baja confianza (0.60 < 0.75)
    for _ in range(5):
        cmd, cambio = filtro.filtrar("FORWARD", confianza=0.60)
        assert cmd == "STOP"
        assert not cambio


def test_filtro_seguridad_consenso_tres_fotogramas():
    """Verifica que se requieran exactamente N fotogramas continuos para conmutar estado."""
    filtro = FiltroSeguridad(umbral_confianza=0.70, fotogramas_confirmacion=3)

    # Cuadro 1 con alta confianza
    cmd1, cambio1 = filtro.filtrar("FORWARD", confianza=0.85)
    assert cmd1 == "STOP"
    assert not cambio1

    # Cuadro 2
    cmd2, cambio2 = filtro.filtrar("FORWARD", confianza=0.88)
    assert cmd2 == "STOP"
    assert not cambio2

    # Cuadro 3: Al tercer fotograma consecutivo debe conmutar a FORWARD
    cmd3, cambio3 = filtro.filtrar("FORWARD", confianza=0.92)
    assert cmd3 == "FORWARD"
    assert cambio3

    # Cuadro 4: Manteniendo FORWARD
    cmd4, cambio4 = filtro.filtrar("FORWARD", confianza=0.90)
    assert cmd4 == "FORWARD"
    assert not cambio4  # Sin cambio de estado


def test_filtro_parada_inmediata():
    """Verifica la parada de emergencia del filtro."""
    filtro = FiltroSeguridad(fotogramas_confirmacion=1)
    filtro.filtrar("FORWARD", confianza=0.99)
    assert filtro.comando_actual == "FORWARD"

    filtro.parada_inmediata()
    assert filtro.comando_actual == "STOP"
    assert len(filtro.historial_predicciones) == 0


def test_simulador_robot_ciclo_basico():
    """Verifica el flujo del simulador virtual."""
    sim = SimuladorRobot()
    assert not sim.conectado
    assert not sim.enviar_comando("FORWARD")  # Falla si no está conectado

    sim.conectar()
    assert sim.conectado
    assert sim.enviar_comando("FORWARD")
    assert sim.ultimo_comando == "FORWARD"

    sim.parada_emergencia()
    assert sim.ultimo_comando == "STOP"

    sim.desconectar()
    assert not sim.conectado
