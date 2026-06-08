"""
Projeto Computacional #3 - Pêndulo Simples (SME0104 - ICMC/USP)
================================================================
Resolução do Problema de Valor de Contorno (PVC) não-linear via
Método de Diferenças Finitas + Newton-Raphson.

Equação governante:
    θ''(t) + (g/L) * sin(θ(t)) = 0,   t ∈ (0, T)
    θ(0) = α,   θ(T) = β

Discretização (diferenças finitas centradas, passo h = T/(m+1)):
    G_i(θ) = θ_{i-1} - 2θ_i + θ_{i+1} + h²*(g/L)*sin(θ_i) = 0
    (com θ_0 = α e θ_{m+1} = β como condições de contorno)

Jacobiana (matriz tridiagonal):
    J_{i,i}   = -2 + h²*(g/L)*cos(θ_i)
    J_{i,i-1} = 1   (subdiagonal)
    J_{i,i+1} = 1   (superdiagonal)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve_banded

# ─────────────────────────────────────────────
# 1. PARÂMETROS FÍSICOS E COMPUTACIONAIS
# ─────────────────────────────────────────────
T     = 2 * np.pi   # Intervalo de tempo
L     = 1.0         # Comprimento da haste (m)
g     = 9.8         # Gravidade Terra (m/s²)
g_mer = 3.7         # Gravidade Mercúrio (m/s²)
alpha = 0.7         # θ(0)  (rad)
beta  = 0.7         # θ(T)  (rad)
TOL   = 1e-10       # Tolerância para convergência
MAXITER = 100       # Número máximo de iterações


# ─────────────────────────────────────────────
# 2. MONTAGEM DO SISTEMA NÃO-LINEAR G(θ)
# ─────────────────────────────────────────────
def montar_sistema(theta, h, L, g, alpha, beta):
    """
    Computa o vetor resíduo G(θ) ∈ R^m.

    G_i = θ_{i-1} - 2*θ_i + θ_{i+1} + h²*(g/L)*sin(θ_i)

    Os nós de contorno (θ_0 = alpha, θ_{m+1} = beta)
    são incorporados nos termos i=0 e i=m-1.
    """
    m  = len(theta) # quantidade de passos
    h2 = h ** 2
    G  = np.zeros(m)

    for i in range(m):

        # tratando as condições de contorno do problema:
        if i == 0: 
            theta_prev = alpha   
        else:
            theta_prev = theta[i - 1]

        if i == m-1:
            theta_next = beta
        else:
            theta_next = theta[i + 1]

        # Aplicando o método das diferenças finitas na EDO:
        G[i] = theta_prev - 2*theta[i] + theta_next + h2 * (g/L) * np.sin(theta[i])

    return G


# ─────────────────────────────────────────────
# 3. JACOBIANA ANALÍTICA (TRIDIAGONAL)
# ─────────────────────────────────────────────
def calcular_jacobiana(theta, h, L, g):
    """
    Monta J(θ) ∈ R^{m×m} (matriz tridiagonal).

    Diagonal principal:  J_{i,i}   = -2 + h²*(g/L)*cos(θ_i)
    Sub/superdiagonal:   J_{i,i±1} = 1
    """
    m  = len(theta)
    h2 = h ** 2
    J  = np.zeros((m, m))

    for i in range(m):
        J[i, i] = -2.0 + h2 * (g/L) * np.cos(theta[i])
        if i > 0:
            J[i, i-1] = 1.0
        if i < m-1:
            J[i, i+1] = 1.0

    return J


# ─────────────────────────────────────────────
# 4. SOLVER NEWTON-RAPHSON
# ─────────────────────────────────────────────
def newton_raphson(theta0, h, L, g, alpha, beta, tol=TOL, maxiter=MAXITER):
    """
    Resolve G(θ) = 0 partindo do chute inicial theta0.

    Retorna:
        theta_sol  : solução convergida
        erros      : histórico do erro relativo por iteração
        n_iter     : número de iterações realizadas
    """
    theta_k = theta0.copy()
    erros   = []

    for k in range(maxiter):
        G = montar_sistema(theta_k, h, L, g, alpha, beta)
        J = calcular_jacobiana(theta_k, h, L, g)

        # Resolve J * Δθ = -G
        delta_theta = np.linalg.solve(J, -G)
        theta_k    += delta_theta

        # Erro relativo  ε_r = ||Δθ|| / ||θ_k||
        norma_delta = np.linalg.norm(delta_theta, ord=np.inf)
        norma_theta = np.linalg.norm(theta_k, ord=np.inf)
        eps_r = norma_delta / (norma_theta + 1e-300)
        erros.append(eps_r)

        if eps_r < tol:
            print(f"  Convergiu em {k+1} iterações  (ε_r = {eps_r:.2e})")
            return theta_k, erros, k+1

    print(f"  AVISO: não convergiu em {maxiter} iterações (ε_r = {erros[-1]:.2e})")
    return theta_k, erros, maxiter


# ─────────────────────────────────────────────
# 5. FUNÇÃO AUXILIAR: SOLUÇÃO COMPLETA
# ─────────────────────────────────────────────
def resolver_pvc(m, L, g, alpha, beta, chute="constante", label=""):
    """
    Monta a malha, escolhe o chute inicial e resolve o PVC.
    Retorna (t_completo, theta_completo, erros).
    """
    h  = T / (m + 1)
    t_int = np.array([(i+1)*h for i in range(m)])   # nós internos

    # ── Chute inicial ──────────────────────────────
    if chute == "constante":
        theta0 = np.full(m, 0.7)
    elif chute == "senoidal":
        theta0 = 0.7 - np.sin(t_int / 2)
    else:
        raise ValueError("chute deve ser 'constante' ou 'senoidal'")

    print(f"\n[{label}]  m={m}, chute={chute}")
    theta_sol, erros, n_iter = newton_raphson(theta0, h, L, g, alpha, beta)

    # Adiciona condições de contorno
    t_full     = np.concatenate([[0], t_int, [T]])
    theta_full = np.concatenate([[alpha], theta_sol, [beta]])

    return t_full, theta_full, erros


# ─────────────────────────────────────────────
# 6. EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── 6.1 Teste com m=100 e m=1000, dois chutes ─────────────────────────
    resultados = {}
    for m in [100, 1000]:
        for chute in ["constante", "senoidal"]:
            key = (m, chute)
            t, theta, erros = resolver_pvc(m, L, g, alpha, beta,
                                           chute=chute,
                                           label=f"m={m}, {chute}")
            resultados[key] = (t, theta, erros)

    # ── 6.2 Gráfico das soluções (m=100) ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pêndulo Simples — Soluções θ(t) via Newton-Raphson", fontsize=14)

    for idx, m in enumerate([100, 1000]):
        ax = axes[idx]
        for chute, ls in [("constante", "-"), ("senoidal", "--")]:
            t, theta, _ = resultados[(m, chute)]
            ax.plot(t, theta, ls, label=f"Chute {chute}")
        ax.set_title(f"m = {m} pontos internos")
        ax.set_xlabel("t  (s)")
        ax.set_ylabel("θ  (rad)")
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("solucoes_theta.png", dpi=150)
    plt.close()

    # ── 6.3 Gráfico de convergência (erro relativo) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Convergência de Newton-Raphson — Erro Relativo ε_r", fontsize=14)

    for idx, m in enumerate([100, 1000]):
        ax = axes[idx]
        for chute, ls, cor in [("constante", "o-", "tab:blue"),
                                ("senoidal",  "s--","tab:orange")]:
            _, _, erros = resultados[(m, chute)]
            ax.semilogy(range(1, len(erros)+1), erros,
                        ls, color=cor, label=f"Chute {chute}", markersize=4)
        ax.set_title(f"m = {m} pontos internos")
        ax.set_xlabel("Iteração k")
        ax.set_ylabel("ε_r  (escala log)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.4)

    plt.tight_layout()
    plt.savefig("convergencia.png", dpi=150)
    plt.close()

    # ── 6.4 Comparação linear (sin θ ≈ θ) vs não-linear ──────────────────
    m = 100
    h = T / (m + 1)
    t_int = np.array([(i+1)*h for i in range(m)])

    def montar_sistema_linear(theta, h, L, g, alpha, beta):
        """Versão linearizada: sin(θ) ≈ θ"""
        m  = len(theta)
        h2 = h ** 2
        G  = np.zeros(m)
        for i in range(m):
            theta_prev = alpha if i == 0   else theta[i-1]
            theta_next = beta  if i == m-1 else theta[i+1]
            G[i] = theta_prev - 2*theta[i] + theta_next + h2*(g/L)*theta[i]
        return G

    def calcular_jacobiana_linear(theta, h, L, g):
        """Jacobiana do modelo linearizado (constante)"""
        m  = len(theta)
        h2 = h ** 2
        J  = np.zeros((m, m))
        for i in range(m):
            J[i, i] = -2.0 + h2*(g/L)
            if i > 0:   J[i, i-1] = 1.0
            if i < m-1: J[i, i+1] = 1.0
        return J

    # Resolve linearizado
    theta0_lin = np.full(m, 0.7)
    J_lin = calcular_jacobiana_linear(theta0_lin, h, L, g)
    G_lin = montar_sistema_linear(theta0_lin, h, L, g, alpha, beta)
    theta_lin = np.linalg.solve(J_lin, -G_lin) + theta0_lin
    # Para o modelo linear a Jacobiana é constante → basta uma iteração
    t_lin = np.concatenate([[0], t_int, [T]])
    theta_lin_full = np.concatenate([[alpha], theta_lin, [beta]])

    t_nl, theta_nl, _ = resolver_pvc(m, L, g, alpha, beta,
                                     chute="constante", label="Não-linear (ref)")

    plt.figure(figsize=(8, 5))
    plt.plot(t_nl,  theta_nl,       label="Não-linear  sin(θ)")
    plt.plot(t_lin, theta_lin_full, "--", label="Linear  sin(θ) ≈ θ")
    plt.title("Comparação: modelo não-linear vs linearizado  (m=100)")
    plt.xlabel("t  (s)"); plt.ylabel("θ  (rad)")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("linear_vs_naolinear.png", dpi=150)
    plt.close()

    # ── 6.5 Estudo paramétrico: Terra vs Mercúrio ─────────────────────────
    plt.figure(figsize=(8, 5))
    for grav, nome, ls in [(g, "Terra (g=9.8)", "-"),
                           (g_mer, "Mercúrio (g=3.7)", "--")]:
        t_p, th_p, _ = resolver_pvc(m, L, grav, alpha, beta,
                                    chute="constante", label=nome)
        plt.plot(t_p, th_p, ls, label=nome)
    plt.title("Estudo Paramétrico: Terra vs Mercúrio")
    plt.xlabel("t  (s)"); plt.ylabel("θ  (rad)")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("terra_vs_mercurio.png", dpi=150)
    plt.close()

    # ── 6.6 Animação comparativa (Grid): Terra vs Mercúrio ────────────────
    # Recupera os dados da Terra (m=100, chute constante)
    t_anim, theta_anim_terra, _ = resultados[(100, "constante")]

    # Calcula os dados para Mercúrio usando os mesmos parâmetros de malha
    _, theta_anim_mercurio, _ = resolver_pvc(100, L, g_mer, alpha, beta,
                                             chute="constante", label="Mercúrio (Animação)")

    # Configuração da Figura com 2 subplots (1 linha, 2 colunas)
    fig_an, (ax_terra, ax_mer) = plt.subplots(1, 2, figsize=(10, 5))
    fig_an.suptitle("Comparação Dinâmica: Pêndulo Simples", fontsize=14)

    # Ajustes visuais para ambos os gráficos
    for ax in (ax_terra, ax_mer):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    ax_terra.set_title("Terra (g = 9.8 m/s²)")
    ax_mer.set_title("Mercúrio (g = 3.7 m/s²)")

    # Elementos visuais - Terra (Vermelho)
    haste_t, = ax_terra.plot([], [], "k-",  lw=2)
    bola_t,  = ax_terra.plot([], [], "ro",  ms=14)
    traj_t,  = ax_terra.plot([], [], "r--", lw=0.8, alpha=0.5)
    tempo_txt_t = ax_terra.text(0.02, 0.95, "", transform=ax_terra.transAxes)

    # Elementos visuais - Mercúrio (Ciano)
    haste_m, = ax_mer.plot([], [], "k-",  lw=2)
    bola_m,  = ax_mer.plot([], [], "co",  ms=14)
    traj_m,  = ax_mer.plot([], [], "c--", lw=0.8, alpha=0.5)
    tempo_txt_m = ax_mer.text(0.02, 0.95, "", transform=ax_mer.transAxes)

    # Listas para rastrear as trajetórias
    xs_t, ys_t = [], []
    xs_m, ys_m = [], []

    def init():
        haste_t.set_data([], [])
        bola_t.set_data([], [])
        traj_t.set_data([], [])
        haste_m.set_data([], [])
        bola_m.set_data([], [])
        traj_m.set_data([], [])
        xs_t.clear(); ys_t.clear()
        xs_m.clear(); ys_m.clear()
        return haste_t, bola_t, traj_t, tempo_txt_t, haste_m, bola_m, traj_m, tempo_txt_m

    def update(frame):
        # Atualização Terra
        th_t = theta_anim_terra[frame]
        x_t  =  L * np.sin(th_t)
        y_t  = -L * np.cos(th_t)
        haste_t.set_data([0, x_t], [0, y_t])
        bola_t.set_data([x_t], [y_t])
        xs_t.append(x_t); ys_t.append(y_t)
        traj_t.set_data(xs_t, ys_t)
        tempo_txt_t.set_text(f"t = {t_anim[frame]:.2f} s")

        # Atualização Mercúrio
        th_m = theta_anim_mercurio[frame]
        x_m  =  L * np.sin(th_m)
        y_m  = -L * np.cos(th_m)
        haste_m.set_data([0, x_m], [0, y_m])
        bola_m.set_data([x_m], [y_m])
        xs_m.append(x_m); ys_m.append(y_m)
        traj_m.set_data(xs_m, ys_m)
        tempo_txt_m.set_text(f"t = {t_anim[frame]:.2f} s")

        return haste_t, bola_t, traj_t, tempo_txt_t, haste_m, bola_m, traj_m, tempo_txt_m

    # Gerar e salvar a animação conjunta
    ani = animation.FuncAnimation(fig_an, update, frames=len(t_anim),
                                  init_func=init, interval=80, blit=True)

    caminho_gif = "pendulo_animacao_comparativa.gif"
    ani.save(caminho_gif, writer="pillow", fps=15)
    plt.close()

    print("\n✓ Todos os arquivos gerados em ")
    print("  • pendulo_simples.py")
    print("  • solucoes_theta.png")
    print("  • convergencia.png")
    print("  • linear_vs_naolinear.png")
    print("  • terra_vs_mercurio.png")
    print(f"  • {caminho_gif.split('/')[-1]}")
