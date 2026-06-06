from manim import *
import numpy as np

# ─────────────────────────────────────────────
# 1. PARÂMETROS NUMÉRICOS E FÍSICOS
# ─────────────────────────────────────────────
T_MAX = 2 * np.pi
L = 2.5       # Aumentado para melhor escala visual no canvas do Manim
m = 100
alpha = 0.7
beta = 0.7

def resolver_pvc_simples(g):
    """
    Resolve o PVC usando Newton-Raphson com chute constante.
    Retorna os vetores de tempo e ângulo (t_full, theta_full).
    """
    h = T_MAX / (m + 1)
    t_int = np.array([(i + 1) * h for i in range(m)])
    theta = np.full(m, 0.7)

    for _ in range(100):
        G = np.zeros(m)
        J = np.zeros((m, m))
        h2 = h**2
        for i in range(m):
            tp = alpha if i == 0 else theta[i-1]
            tn = beta if i == m-1 else theta[i+1]
            G[i] = tp - 2*theta[i] + tn + h2*(g/L)*np.sin(theta[i])

            J[i, i] = -2.0 + h2*(g/L)*np.cos(theta[i])
            if i > 0: J[i, i-1] = 1.0
            if i < m-1: J[i, i+1] = 1.0

        delta = np.linalg.solve(J, -G)
        theta += delta
        if np.linalg.norm(delta) < 1e-10:
            break

    t_full = np.concatenate([[0], t_int, [T_MAX]])
    theta_full = np.concatenate([[alpha], theta, [beta]])
    return t_full, theta_full

# ─────────────────────────────────────────────
# 2. CENA DA ANIMAÇÃO NO MANIM
# ─────────────────────────────────────────────
class ComparacaoPendulos(Scene):
    def construct(self):
        # Textos e Títulos minimalistas
        titulo = Text("Projeto #3: Pêndulo Simples Não-Linear", font_size=32, weight=BOLD).to_edge(UP)
        self.add(titulo)

        # Cálculo prévio das soluções numéricas
        t_array, theta_terra = resolver_pvc_simples(9.8)
        _, theta_mercurio = resolver_pvc_simples(3.7)

        # O ValueTracker será o nosso "motor de tempo" contínuo para a animação
        tempo = ValueTracker(0)

        # Função encapsulada para criar e animar cada pêndulo
        def criar_pendulo(origem, cor_haste, cor_massa, theta_array, label_nome):
            pivo = Dot(origem, radius=0.06, color=WHITE)
            haste = Line(origem, origem + DOWN * L, color=cor_haste, stroke_width=3)
            massa = Dot(haste.get_end(), radius=0.18, color=cor_massa)

            # Efeito de rastro que desaparece gradualmente (Dissipating Trail)
            rastro = TracedPath(massa.get_center, stroke_color=cor_massa, stroke_width=4, dissipating_time=0.4)

            # Rótulo de identificação
            label = Text(label_nome, font_size=24, color=cor_massa).next_to(pivo, UP, buff=0.3)

            # A função Updater amarra a física aos objetos na tela iterativamente
            def atualizar_pendulo(mob):
                t_atual = tempo.get_value()
                # Interpolação para garantir fluidez mesmo entre os pontos da malha (m=100)
                th = np.interp(t_atual, t_array, theta_array)

                # Cálculo vetorial da nova posição (0 radianos = eixo y negativo / DOWN)
                nova_pos = origem + L * np.array([np.sin(th), -np.cos(th), 0])

                mob[0].put_start_and_end_on(origem, nova_pos) # Atualiza a haste
                mob[1].move_to(nova_pos)                      # Atualiza a massa

            # Agrupa haste e massa e anexa a função de atualização
            grupo_dinamico = VGroup(haste, massa)
            grupo_dinamico.add_updater(atualizar_pendulo)

            return VGroup(pivo, label, rastro, grupo_dinamico)

        # Instanciação dos sistemas lado a lado
        pendulo_t = criar_pendulo(LEFT * 3.5 + UP, BLUE_D, BLUE_C, theta_terra, "Terra (g = 9.8)")
        pendulo_m = criar_pendulo(RIGHT * 3.5 + UP, RED_E, ORANGE, theta_mercurio, "Mercúrio (g = 3.7)")

        # Relógio dinâmico no canto inferior direito
        relogio = DecimalNumber(0, num_decimal_places=2, font_size=28).to_corner(DR)
        label_relogio = Text("Tempo (s): ", font_size=28).next_to(relogio, LEFT)
        relogio.add_updater(lambda d: d.set_value(tempo.get_value()))

        # Adiciona todos os elementos estáticos e iniciais à tela
        self.add(pendulo_t, pendulo_m, label_relogio, relogio)

        # ─────────────────────────────────────────────
        # 3. RENDERIZAÇÃO DO MOVIMENTO
        # ─────────────────────────────────────────────
        # Evolui o tempo de 0 até T_MAX. Rate_func=linear garante que a física
        # corra no seu tempo natural, sem acelerações artificiais de animação.
        self.play(
            tempo.animate.set_value(T_MAX),
            run_time=6,  # Duração real do vídeo em segundos
            rate_func=linear
        )

        # Mantém a tela parada no frame final por 1 segundo
        self.wait(1)
