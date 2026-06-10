"""
Projeto Computacional #3 — Pêndulo Simples (SME0104 - ICMC/USP)
================================================================
Resolução do Problema de Valor de Contorno (PVC) não-linear via
Método de Diferenças Finitas Centradas + Newton-Raphson.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─────────────────────────────────────────────
# 1. PARÂMETROS FÍSICOS E COMPUTACIONAIS
# ─────────────────────────────────────────────
T       = 2 * np.pi   # Período de análise [s]
L       = 1.0         # Comprimento do pêndulo [m]
g_terra = 9.8         # Aceleração gravitacional — Terra [m/s²]
g_mer   = 3.7         # Aceleração gravitacional — Mercúrio [m/s²]
alpha   = 0.7         # Condição de contorno: θ(0) [rad]
beta    = 0.7         # Condição de contorno: θ(T) [rad]
TOL     = 1e-10       # Tolerância de convergência
MAXITER = 100         # Número máximo de iterações de Newton


# ─────────────────────────────────────────────
# 2. RESÍDUO  G(θ)  — sistema não linear
# ─────────────────────────────────────────────
def montar_residuo(theta, h, L, g, alpha, beta):
    """
    Calcula o vetor resíduo G(θ) ∈ R^m conforme a discretização
    de diferenças finitas centradas da equação do pêndulo.

    Para i = 1, …, m  (índice base-0: i = 0, …, m-1):
        G_i = θ_{i-1} - 2·θ_i + θ_{i+1} + h²·(g/L)·sin(θ_i)

    Condições de contorno:
        θ_{-1}  ≡ α   (nó i = 0)
        θ_{m}   ≡ β   (nó i = m-1)
    """
    m  = len(theta)
    h2 = h * h
    G  = np.empty(m)

    for i in range(m):
        theta_esq = alpha       if i == 0   else theta[i - 1]
        theta_dir = beta        if i == m-1 else theta[i + 1]
        G[i] = theta_esq - 2.0 * theta[i] + theta_dir \
               + h2 * (g / L) * np.sin(theta[i])
    return G


# ─────────────────────────────────────────────
# 3. JACOBIANA  J(θ)  — matriz tridiagonal
# ─────────────────────────────────────────────
def montar_jacobiana(theta, h, L, g):
    """
    Monta a Jacobiana analítica J(θ) ∈ R^{m×m}.

    Estrutura tridiagonal:
        Diagonal principal : J_{i,i}   = -2 + h²·(g/L)·cos(θ_i)
        Subdiagonal        : J_{i,i-1} = 1
        Superdiagonal      : J_{i,i+1} = 1

    Nota: por ser tridiagonal, seria possível aplicar o Algoritmo de
    Thomas (eliminação de Gauss especializada para sistemas tridiagonais,
    complexidade O(m)) para resolver J·s = -G de forma eficiente.
    Nesta implementação utilizamos np.linalg.solve (O(m³)) para manter
    clareza pedagógica e fidelidade ao algoritmo do livro-texto.
    """
    m  = len(theta)
    h2 = h * h
    J  = np.zeros((m, m))

    for i in range(m):
        J[i, i] = -2.0 + h2 * (g / L) * np.cos(theta[i])
        if i > 0:
            J[i, i - 1] = 1.0
        if i < m - 1:
            J[i, i + 1] = 1.0
    return J


# ─────────────────────────────────────────────
# 4. MÉTODO DE NEWTON-RAPHSON
#    (Ruggiero & Lopes, Algoritmo — Seção 4.2)
# ─────────────────────────────────────────────
def newton_raphson(theta0, h, L, g, alpha, beta, tol=TOL, maxiter=MAXITER):
    """
    Resolve G(θ) = 0 pelo Método de Newton (Ruggiero & Lopes, Cap. 4).

    Algoritmo:
        Passo 1 : Calcule G(θ^(k)) e J(θ^(k))
        Passo 2 : Se ||G(θ^(k))||_∞ < ε₁ → pare  (critério no resíduo)
        Passo 3 : Resolva J(θ^(k)) · s^(k) = -G(θ^(k))
        Passo 4 : θ^(k+1) = θ^(k) + s^(k)
        Passo 5 : Se ||s^(k)||_∞ / ||θ^(k+1)||_∞ < ε₂ → pare
        Passo 6 : k ← k + 1; volte ao Passo 1

    Retorna
    -------
    theta_k : solução convergida
    erros   : histórico do erro relativo ε_r por iteração
    n_iter  : número de iterações realizadas
    """
    theta_k = theta0.copy()
    erros   = []

    for k in range(maxiter):
        # ── Passo 1: avalia G e J ────────────────────────────────────────
        G = montar_residuo(theta_k, h, L, g, alpha, beta)
        J = montar_jacobiana(theta_k, h, L, g)

        # ── Passo 2: critério no resíduo ─────────────────────────────────
        if np.linalg.norm(G, ord=np.inf) < tol:
            print(f"  Convergiu (resíduo) em {k} iterações")
            return theta_k, erros, k

        # ── Passo 3: resolve o sistema linear J · s = -G ─────────────────
        s = np.linalg.solve(J, -G)

        # ── Passo 4: atualiza a iterada ───────────────────────────────────
        theta_k = theta_k + s

        # ── Passo 5: critério no passo (erro relativo) ────────────────────
        eps_r = np.linalg.norm(s, ord=np.inf) \
                / (np.linalg.norm(theta_k, ord=np.inf) + 1e-300)
        erros.append(eps_r)

        if eps_r < tol:
            print(f"  Convergiu em {k + 1} iterações  (ε_r = {eps_r:.2e})")
            return theta_k, erros, k + 1

        # ── Passo 6: próxima iteração ─────────────────────────────────────

    print(f"  AVISO: não convergiu em {maxiter} iterações"
          f"  (ε_r = {erros[-1]:.2e})")
    return theta_k, erros, maxiter


# ─────────────────────────────────────────────
# 5. SOLVER COMPLETO DO PVC
# ─────────────────────────────────────────────
def resolver_pvc(m, L, g, alpha, beta, chute="constante", label=""):
    """
    Resolve o PVC do pêndulo com m pontos internos.

    Parâmetros
    ----------
    m      : número de pontos internos da malha
    chute  : 'constante' → θ^(0) = (0.7, …, 0.7)
             'senoidal'  → (θ^(0))_i = 0.7 − sin(t_i / 2)

    Retorna (t_completo, theta_completo, erros)
    """
    h     = T / (m + 1)
    t_int = np.array([(i + 1) * h for i in range(m)])   # nós internos

    if chute == "constante":
        theta0 = np.full(m, 0.7)
    elif chute == "senoidal":
        theta0 = 0.7 - np.sin(t_int / 2)
    else:
        raise ValueError("chute deve ser 'constante' ou 'senoidal'")

    print(f"\n[{label}]  m = {m},  chute = {chute}")
    theta_sol, erros, n_iter = newton_raphson(theta0, h, L, g, alpha, beta)

    t_full     = np.concatenate([[0.0],  t_int,     [T]])
    theta_full = np.concatenate([[alpha], theta_sol, [beta]])
    return t_full, theta_full, erros


# ─────────────────────────────────────────────
# 6. MODELO LINEARIZADO  sin(θ) ≈ θ
# ─────────────────────────────────────────────
def resolver_pvc_linear(m, L, g, alpha, beta):
    """
    Resolve a versão linearizada (sin θ ≈ θ).
    A Jacobiana é constante → basta uma iteração de Newton.
    """
    h     = T / (m + 1)
    t_int = np.array([(i + 1) * h for i in range(m)])
    h2    = h * h

    # Monta sistema tridiagonal linear:  A·θ = b
    A = np.zeros((m, m))
    b = np.zeros(m)
    for i in range(m):
        A[i, i] = -2.0 + h2 * (g / L)
        if i > 0:
            A[i, i - 1] = 1.0
        if i < m - 1:
            A[i, i + 1] = 1.0
        # termos das condições de contorno no vetor b
        if i == 0:
            b[i] = -alpha
        if i == m - 1:
            b[i] -= beta

    theta_lin = np.linalg.solve(A, b)

    t_full     = np.concatenate([[0.0],  t_int,     [T]])
    theta_full = np.concatenate([[alpha], theta_lin, [beta]])
    return t_full, theta_full


# ─────────────────────────────────────────────
# 7. ESTUDO DE SENSIBILIDADE (fractais)
# ─────────────────────────────────────────────
def mapa_sensibilidade(m=80, n_alpha=60, n_beta=60):
    """
    Varre um grid de condições iniciais (alpha, beta) e registra
    o número de iterações de Newton necessário para convergir.
    Regiões de alta variação revelam estrutura fractal na bacia
    de atração do método.
    """
    alphas = np.linspace(0.1, 2.5, n_alpha)
    betas  = np.linspace(0.1, 2.5, n_beta)
    itermap = np.zeros((n_beta, n_alpha))

    for j, a in enumerate(alphas):
        for i, b_ in enumerate(betas):
            h      = T / (m + 1)
            theta0 = np.full(m, (a + b_) / 2)
            try:
                _, _, n = newton_raphson(theta0, h, L, g_terra, a, b_,
                                         tol=1e-8, maxiter=50)
                itermap[i, j] = n
            except np.linalg.LinAlgError:
                itermap[i, j] = 50  # singular → divergiu

    return alphas, betas, itermap


# ═══════════════════════════════════════════════════════════════════
# 8. EXECUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 8.1  Solução para m = 100 e m = 1000, dois chutes ─────────
    resultados = {}
    for m in [100, 1000]:
        for chute in ["constante", "senoidal"]:
            t, theta, erros = resolver_pvc(m, L, g_terra, alpha, beta,
                                           chute=chute,
                                           label=f"m={m}, {chute}")
            resultados[(m, chute)] = (t, theta, erros)

    # ── 8.2  Gráfico das soluções ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pêndulo Simples — Solução θ(t) via Newton-Raphson", fontsize=14)

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
    print("Salvo: solucoes_theta.png")

    # ── 8.3  Convergência (erro relativo em escala log) ────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Convergência de Newton-Raphson — Erro Relativo ε_r", fontsize=14)

    for idx, m in enumerate([100, 1000]):
        ax = axes[idx]
        for chute, ls, cor in [("constante", "o-", "tab:blue"),
                                ("senoidal",  "s--", "tab:orange")]:
            _, _, erros = resultados[(m, chute)]
            ax.semilogy(range(1, len(erros) + 1), erros,
                        ls, color=cor, label=f"Chute {chute}", markersize=5)
        ax.set_title(f"m = {m} pontos internos")
        ax.set_xlabel("Iteração k")
        ax.set_ylabel("ε_r  (escala log)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.4)

    plt.tight_layout()
    plt.savefig("convergencia.png", dpi=150)
    plt.close()
    print("Salvo: convergencia.png")

    # ── 8.4  Não-linear vs Linearizado ────────────────────────────
    m = 100
    t_nl, theta_nl, _ = resolver_pvc(m, L, g_terra, alpha, beta,
                                     chute="senoidal", label="Não-linear")
    t_lin, theta_lin = resolver_pvc_linear(m, L, g_terra, alpha, beta)

    plt.figure(figsize=(8, 5))
    plt.plot(t_nl,  theta_nl,  label="Não-linear  [sin(θ)]")
    plt.plot(t_lin, theta_lin, "--", label="Linearizado  [sin(θ) ≈ θ]")
    plt.title("Comparação: modelo não-linear vs linearizado  (m = 100, α = β = 0.7 rad)")
    plt.xlabel("t  (s)")
    plt.ylabel("θ  (rad)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("linear_vs_naolinear.png", dpi=150)
    plt.close()
    print("Salvo: linear_vs_naolinear.png")

    # ── 8.5  Estudo paramétrico: Terra vs Mercúrio ─────────────────
    m = 100
    plt.figure(figsize=(8, 5))
    for grav, nome, ls, cor in [
        (g_terra, f"Terra  (g = {g_terra} m/s²)", "-",  "tab:blue"),
        (g_mer,   f"Mercúrio  (g = {g_mer} m/s²)", "--", "tab:red"),
    ]:
        t_p, th_p, _ = resolver_pvc(m, L, grav, alpha, beta,
                                    chute="constante", label=nome)
        plt.plot(t_p, th_p, ls, color=cor, label=nome)

    plt.title("Estudo Paramétrico: Terra vs Mercúrio  (α = β = 0.7 rad)")
    plt.xlabel("t  (s)")
    plt.ylabel("θ  (rad)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("terra_vs_mercurio.png", dpi=150)
    plt.close()
    print("Salvo: terra_vs_mercurio.png")

    '''
    # ── 8.6  Mapa de sensibilidade / estrutura fractal ─────────────
    print("\nGerando mapa de sensibilidade (fractais)... aguarde.")
    alphas, betas, itermap = mapa_sensibilidade(m=60, n_alpha=80, n_beta=80)

    plt.figure(figsize=(7, 6))
    plt.imshow(itermap, origin="lower", aspect="auto",
               extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
               cmap="inferno")
    cb = plt.colorbar()
    cb.set_label("Iterações de Newton")
    plt.title("Mapa de Sensibilidade: iterações × (α, β)\n"
              "Estrutura fractal na bacia de atração")
    plt.xlabel("α  (rad)")
    plt.ylabel("β  (rad)")
    plt.tight_layout()
    plt.savefig("sensibilidade_fractal.png", dpi=150)
    plt.close()
    print("Salvo: sensibilidade_fractal.png")
    '''

    # ── 8.7  Animação comparativa: Terra vs Mercúrio ───────────────
    t_anim   = resultados[(100, "constante")][0]
    th_terra = resultados[(100, "constante")][1]
    _, th_mer, _ = resolver_pvc(100, L, g_mer, alpha, beta,
                                chute="constante", label="Mercúrio anim.")

    fig_an, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(10, 5))
    fig_an.suptitle("Comparação Dinâmica — Pêndulo Simples", fontsize=13)

    for ax, titulo in [(ax_t, f"Terra  (g = {g_terra} m/s²)"),
                        (ax_m, f"Mercúrio  (g = {g_mer} m/s²)")]:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 0.3)
        ax.set_aspect("equal")
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)

    haste_t, = ax_t.plot([], [], "k-",  lw=2)
    bola_t,  = ax_t.plot([], [], "ro",  ms=14)
    traj_t,  = ax_t.plot([], [], "r--", lw=0.8, alpha=0.5)
    txt_t    = ax_t.text(0.02, 0.95, "", transform=ax_t.transAxes, fontsize=9)

    haste_m, = ax_m.plot([], [], "k-",  lw=2)
    bola_m,  = ax_m.plot([], [], "co",  ms=14)
    traj_m,  = ax_m.plot([], [], "c--", lw=0.8, alpha=0.5)
    txt_m    = ax_m.text(0.02, 0.95, "", transform=ax_m.transAxes, fontsize=9)

    xs_t, ys_t, xs_m, ys_m = [], [], [], []

    def init():
        for obj in [haste_t, bola_t, traj_t, haste_m, bola_m, traj_m]:
            obj.set_data([], [])
        xs_t.clear(); ys_t.clear(); xs_m.clear(); ys_m.clear()
        return haste_t, bola_t, traj_t, txt_t, haste_m, bola_m, traj_m, txt_m

    def update(frame):
        th = th_terra[frame]
        xt, yt = L * np.sin(th), -L * np.cos(th)
        haste_t.set_data([0, xt], [0, yt])
        bola_t.set_data([xt], [yt])
        xs_t.append(xt); ys_t.append(yt)
        traj_t.set_data(xs_t, ys_t)
        txt_t.set_text(f"t = {t_anim[frame]:.2f} s")

        th = th_mer[frame]
        xm, ym = L * np.sin(th), -L * np.cos(th)
        haste_m.set_data([0, xm], [0, ym])
        bola_m.set_data([xm], [ym])
        xs_m.append(xm); ys_m.append(ym)
        traj_m.set_data(xs_m, ys_m)
        txt_m.set_text(f"t = {t_anim[frame]:.2f} s")

        return haste_t, bola_t, traj_t, txt_t, haste_m, bola_m, traj_m, txt_m

    ani = animation.FuncAnimation(fig_an, update, frames=len(t_anim),
                                  init_func=init, interval=80, blit=True)
    ani.save("pendulo_animacao_comparativa.gif", writer="pillow", fps=15)
    plt.close()
    print("Salvo: pendulo_animacao_comparativa.gif")

    print("\n✓ Execução concluída. Arquivos gerados:")
    for f in ["pendulo_simples.py", "solucoes_theta.png", "convergencia.png",
              "linear_vs_naolinear.png", "terra_vs_mercurio.png",
              "sensibilidade_fractal.png", "pendulo_animacao_comparativa.gif"]:
        print(f"  • {f}")
