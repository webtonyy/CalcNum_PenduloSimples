#!/usr/bin/env python3
"""
Aplicacao offline para o Projeto 03 - Pendulo simples.

Resolve o PVC nao linear por diferencas finitas e Newton-Raphson usando norma
infinita no criterio de parada. A interface Tkinter permite alterar g, L, T,
alpha, beta e m e ver graficos/animações atualizados.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk


EPS = 1e-14


def norm_inf(values: list[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def solve_tridiagonal(
    lower: list[float], diag: list[float], upper: list[float], rhs: list[float]
) -> list[float]:
    n = len(diag)
    c = upper[:]
    d = diag[:]
    b = rhs[:]

    for i in range(1, n):
        if abs(d[i - 1]) < EPS:
            raise ArithmeticError("pivo numerico muito pequeno")
        factor = lower[i - 1] / d[i - 1]
        d[i] -= factor * c[i - 1]
        b[i] -= factor * b[i - 1]

    if abs(d[-1]) < EPS:
        raise ArithmeticError("pivo numerico muito pequeno")

    x = [0.0] * n
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        if abs(d[i]) < EPS:
            raise ArithmeticError("pivo numerico muito pequeno")
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    return x


@dataclass
class Params:
    g: float = 9.8
    L: float = 1.0
    T: float = 2.0 * math.pi
    alpha: float = 0.7
    beta: float = 0.7
    m: int = 120
    tol: float = 1e-10
    max_iter: int = 80


@dataclass
class Solution:
    t: list[float]
    theta: list[float]
    interior: list[float]
    h: float
    errors: list[float]
    residuals: list[float]
    converged: bool
    message: str

    @property
    def iterations(self) -> int:
        return len(self.errors)

    @property
    def max_abs_theta(self) -> float:
        return norm_inf(self.theta)


class PendulumAPI:
    """API numerica usada pela interface."""

    @staticmethod
    def mesh(params: Params) -> tuple[list[float], float]:
        m = max(5, min(2000, int(params.m)))
        h = params.T / (m + 1)
        return [i * h for i in range(m + 2)], h

    @staticmethod
    def residual(theta: list[float], params: Params, h: float, nonlinear: bool) -> list[float]:
        m = len(theta)
        scale = 1.0 / (h * h)
        coef = params.g / params.L
        values = [0.0] * m
        for i in range(m):
            left = params.alpha if i == 0 else theta[i - 1]
            right = params.beta if i == m - 1 else theta[i + 1]
            force = math.sin(theta[i]) if nonlinear else theta[i]
            values[i] = scale * (left - 2.0 * theta[i] + right) + coef * force
        return values

    @staticmethod
    def jacobian(theta: list[float], params: Params, h: float, nonlinear: bool) -> tuple[list[float], list[float], list[float]]:
        m = len(theta)
        scale = 1.0 / (h * h)
        coef = params.g / params.L
        lower = [scale] * max(0, m - 1)
        upper = [scale] * max(0, m - 1)
        diag = [
            -2.0 * scale + coef * (math.cos(value) if nonlinear else 1.0)
            for value in theta
        ]
        return lower, diag, upper

    @staticmethod
    def initial_guess(kind: str, t: list[float], params: Params) -> list[float]:
        theta = []
        for i in range(1, len(t) - 1):
            if kind == "sine":
                theta.append(0.7 - math.sin(t[i] / 2.0))
            else:
                theta.append(0.7)
        return theta

    @staticmethod
    def solve_newton(params: Params, kind: str) -> Solution:
        t, h = PendulumAPI.mesh(params)
        theta = PendulumAPI.initial_guess(kind, t, params)
        errors: list[float] = []
        residuals: list[float] = []
        converged = False
        message = "maximo de iteracoes atingido"

        for _ in range(params.max_iter):
            gvec = PendulumAPI.residual(theta, params, h, nonlinear=True)
            lower, diag, upper = PendulumAPI.jacobian(theta, params, h, nonlinear=True)
            step = solve_tridiagonal(lower, diag, upper, [-v for v in gvec])
            nxt = [value + delta for value, delta in zip(theta, step)]
            err = norm_inf(step) / max(norm_inf(nxt), EPS)
            errors.append(err)
            residuals.append(norm_inf(PendulumAPI.residual(nxt, params, h, nonlinear=True)))
            theta = nxt
            if err <= params.tol:
                converged = True
                message = "convergente"
                break

        return Solution(t, [params.alpha, *theta, params.beta], theta, h, errors, residuals, converged, message)

    @staticmethod
    def solve_linearized(params: Params) -> Solution:
        t, h = PendulumAPI.mesh(params)
        m = len(t) - 2
        theta0 = [0.0] * m
        lower, diag, upper = PendulumAPI.jacobian(theta0, params, h, nonlinear=False)
        rhs = [0.0] * m
        scale = 1.0 / (h * h)
        rhs[0] -= scale * params.alpha
        rhs[-1] -= scale * params.beta
        theta = solve_tridiagonal(lower, diag, upper, rhs)
        return Solution(t, [params.alpha, *theta, params.beta], theta, h, [], [], True, "linearizada")

    @staticmethod
    def solve(params: Params) -> dict[str, Solution]:
        return {
            "constant": PendulumAPI.solve_newton(params, "constant"),
            "sine": PendulumAPI.solve_newton(params, "sine"),
            "linear": PendulumAPI.solve_linearized(params),
        }

    @staticmethod
    def sample(solution: Solution, time_value: float) -> float:
        total = solution.t[-1]
        wrapped = time_value % total
        i = min(len(solution.theta) - 1, max(0, round(wrapped / solution.h)))
        return solution.theta[i]

    @staticmethod
    def max_difference(a: Solution, b: Solution) -> float:
        return max(abs(x - y) for x, y in zip(a.theta, b.theta))


class PendulumApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pêndulo simples")
        self.geometry("1180x780")
        self.minsize(980, 680)

        self.result: dict[str, Solution] | None = None
        self.mercury_result: dict[str, Solution] | None = None
        self.elapsed = 0.0
        self.playing = True
        self.pending_update: str | None = None
        self.pending_redraw: str | None = None

        self.vars = {
            "g": tk.DoubleVar(value=9.8),
            "L": tk.DoubleVar(value=1.0),
            "T": tk.DoubleVar(value=2.0 * math.pi),
            "alpha": tk.DoubleVar(value=0.7),
            "beta": tk.DoubleVar(value=0.7),
            "m": tk.IntVar(value=120),
            "tol": tk.DoubleVar(value=1e-10),
            "speed": tk.DoubleVar(value=1.0),
            "choice": tk.StringVar(value="chute senoidal"),
        }

        self.status_var = tk.StringVar(value="calculando...")
        self._build_ui()
        self.recompute()
        self.after(16, self.animate)

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg="#eef2ed")
        style.configure("TFrame", background="#eef2ed")
        style.configure("Sidebar.TFrame", background="#f8faf7")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("TLabel", background="#eef2ed", foreground="#1f2823")
        style.configure("Sidebar.TLabel", background="#f8faf7", foreground="#1f2823")
        style.configure("Panel.TLabel", background="#ffffff", foreground="#1f2823")
        style.configure("TButton", background="#e3efec", foreground="#173b35", padding=6)
        style.map("TButton", background=[("active", "#d5e8e4")])
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#1f2823")
        style.configure("TCombobox", fieldbackground="#ffffff", foreground="#1f2823")
        style.configure("TNotebook", background="#eef2ed", borderwidth=0)
        style.configure("TNotebook.Tab", background="#dde6dc", foreground="#1f2823", padding=(14, 8))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        sidebar_shell = ttk.Frame(root, width=260, style="Sidebar.TFrame")
        sidebar_shell.pack(side="left", fill="y", padx=10, pady=10)
        sidebar_shell.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_shell, bg="#f8faf7", highlightthickness=0)
        sidebar_scroll = ttk.Scrollbar(sidebar_shell, orient="vertical", command=sidebar_canvas.yview)
        sidebar = ttk.Frame(sidebar_canvas, style="Sidebar.TFrame")
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scroll.pack(side="right", fill="y")
        sidebar.bind(
            "<Configure>",
            lambda _event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.bind(
            "<Configure>",
            lambda event: sidebar_canvas.itemconfigure(sidebar_window, width=event.width),
        )

        main = ttk.Frame(root)
        main.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        ttk.Label(sidebar, text="Pêndulo", font=("Arial", 20, "bold"), style="Sidebar.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(sidebar, textvariable=self.status_var, wraplength=220, style="Sidebar.TLabel").pack(anchor="w", pady=(0, 12))

        self._add_scale(sidebar, "g", "g", 1.0, 25.0, 0.05)
        self._add_scale(sidebar, "L", "L", 0.2, 5.0, 0.05)
        self._add_scale(sidebar, "T", "T", 0.5, 14.0, 0.05)
        self._add_entry(sidebar, "α (rad)", "alpha")
        self._add_entry(sidebar, "β (rad)", "beta")
        self._add_entry(sidebar, "m", "m")
        self._add_entry(sidebar, "tol", "tol")
        self._add_scale(sidebar, "vel.", "speed", 0.1, 4.0, 0.1, recompute=False)

        ttk.Label(sidebar, text="animação", style="Sidebar.TLabel").pack(anchor="w", pady=(10, 2))
        choice = ttk.Combobox(
            sidebar,
            textvariable=self.vars["choice"],
            values=("chute constante", "chute senoidal"),
            state="readonly",
        )
        choice.pack(fill="x")

        buttons = ttk.Frame(sidebar)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="Terra", command=lambda: self.set_preset("earth")).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(buttons, text="Mercúrio", command=lambda: self.set_preset("mercury")).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(buttons, text="Padrão", command=lambda: self.set_preset("project")).pack(side="left", expand=True, fill="x", padx=(4, 0))

        buttons_2 = ttk.Frame(sidebar)
        buttons_2.pack(fill="x")
        self.play_button = ttk.Button(buttons_2, text="Pausar", command=self.toggle_play)
        self.play_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(buttons_2, text="Reiniciar", command=self.reset_time).pack(side="left", expand=True, fill="x", padx=(4, 0))

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)

        pendulum_tab = ttk.Frame(notebook, style="Panel.TFrame")
        planet_tab = ttk.Frame(notebook, style="Panel.TFrame")
        convergence_tab = ttk.Frame(notebook)
        chutes_tab = ttk.Frame(notebook)
        linear_tab = ttk.Frame(notebook)
        notebook.add(pendulum_tab, text="Pêndulo")
        notebook.add(planet_tab, text="Terra x Mercúrio")
        notebook.add(convergence_tab, text="Convergência")
        notebook.add(chutes_tab, text="Chutes iniciais")
        notebook.add(linear_tab, text="Linearização")

        self.pendulum_canvas = self._canvas_panel(pendulum_tab)
        self.planet_canvas = self._canvas_panel(planet_tab)
        self.error_canvas = self._canvas_panel(convergence_tab)
        self.solution_canvas = self._canvas_panel(chutes_tab)
        self.linear_canvas = self._canvas_panel(linear_tab)

        for canvas in (self.pendulum_canvas, self.planet_canvas, self.error_canvas, self.solution_canvas, self.linear_canvas):
            canvas.bind("<Configure>", lambda _event: self.schedule_static_redraw())

    def _canvas_panel(self, parent: ttk.Frame) -> tk.Canvas:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        canvas = tk.Canvas(frame, bg="#ffffff", highlightthickness=1, highlightbackground="#d8dfd6")
        canvas.pack(fill="both", expand=True, padx=8, pady=8)
        return canvas

    def _add_scale(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        from_: float,
        to: float,
        step: float,
        recompute: bool = True,
    ) -> None:
        ttk.Label(parent, text=label, style="Sidebar.TLabel").pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(parent, style="Sidebar.TFrame")
        row.pack(fill="x")
        scale = ttk.Scale(row, from_=from_, to=to, variable=self.vars[key], command=lambda _v: self.schedule_update(recompute))
        scale.pack(side="left", fill="x", expand=True)
        value = ttk.Label(row, width=8)
        value.pack(side="left", padx=(8, 0))

        def refresh_label(*_args: object) -> None:
            value.config(text=f"{self.vars[key].get():.3g}")

        self.vars[key].trace_add("write", refresh_label)
        refresh_label()

    def _add_entry(self, parent: ttk.Frame, label: str, key: str) -> None:
        ttk.Label(parent, text=label, style="Sidebar.TLabel").pack(anchor="w", pady=(6, 0))
        entry = ttk.Entry(parent, textvariable=self.vars[key])
        entry.pack(fill="x")
        entry.bind("<Return>", lambda _event: self.recompute())
        entry.bind("<FocusOut>", lambda _event: self.recompute())

    def schedule_update(self, recompute: bool = True) -> None:
        if not recompute:
            return
        if self.pending_update is not None:
            self.after_cancel(self.pending_update)
        self.pending_update = self.after(180, self.recompute)

    def schedule_static_redraw(self) -> None:
        if self.pending_redraw is not None:
            self.after_cancel(self.pending_redraw)
        self.pending_redraw = self.after(120, self._static_redraw_now)

    def _static_redraw_now(self) -> None:
        self.pending_redraw = None
        self.draw_all_static()

    def params(self) -> Params:
        return Params(
            g=float(self.vars["g"].get()),
            L=max(0.05, float(self.vars["L"].get())),
            T=max(0.05, float(self.vars["T"].get())),
            alpha=float(self.vars["alpha"].get()),
            beta=float(self.vars["beta"].get()),
            m=max(5, min(2000, int(self.vars["m"].get()))),
            tol=max(1e-14, float(self.vars["tol"].get())),
        )

    def recompute(self) -> None:
        self.pending_update = None
        try:
            params = self.params()
            self.result = PendulumAPI.solve(params)
            mercury_params = Params(**{**params.__dict__, "g": 3.7})
            self.mercury_result = PendulumAPI.solve(mercury_params)
            a = self.result["constant"]
            b = self.result["sine"]
            status = "ok" if a.converged and b.converged else "verificar"
            self.status_var.set(f"{status} · it. {a.iterations}/{b.iterations}")
            self.elapsed = 0.0
            self.draw_all_static()
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"erro: {exc}")

    def draw_all_static(self) -> None:
        if not self.result:
            return
        self.draw_error_chart()
        self.draw_solution_chart()
        self.draw_linear_chart()

    def clear(self, canvas: tk.Canvas) -> tuple[int, int]:
        canvas.update_idletasks()
        width = max(260, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#ffffff", outline="")
        return width, height

    def draw_axes(self, canvas: tk.Canvas, width: int, height: int, ylabel: str) -> tuple[int, int, int, int]:
        left, top, right, bottom = 78, 52, 32, 56
        canvas.create_line(left, top, left, height - bottom, width=1, fill="#d8dfd6")
        canvas.create_line(left, height - bottom, width - right, height - bottom, width=1, fill="#d8dfd6")
        canvas.create_text(22, top + 70, text=ylabel, fill="#667169", angle=90)
        return left, top, right, bottom

    def draw_polyline(
        self,
        canvas: tk.Canvas,
        points: list[tuple[float, float]],
        color: str,
        width: int = 2,
    ) -> None:
        if len(points) >= 2:
            coords = [coord for point in points for coord in point]
            canvas.create_line(*coords, fill=color, width=width, smooth=False)

    def draw_error_chart(self) -> None:
        if not self.result:
            return
        canvas = self.error_canvas
        width, height = self.clear(canvas)
        left, top, right, bottom = self.draw_axes(canvas, width, height, "log10(εᵣ)")
        series = [
            ("constante", self.result["constant"].errors, "#0f766e"),
            ("senoidal", self.result["sine"].errors, "#b04d28"),
        ]
        all_errors = [e for _, values, _ in series for e in values if e > 0.0]
        if not all_errors:
            return
        min_y = math.floor(math.log10(min(all_errors))) - 0.5
        max_y = math.ceil(math.log10(max(all_errors))) + 0.5
        max_iter = max(len(values) for _, values, _ in series)
        for label, values, color in series:
            pts = []
            for i, value in enumerate(values):
                x = left + (i / max(1, max_iter - 1)) * (width - left - right)
                log_value = math.log10(max(value, 1e-16))
                y = height - bottom - ((log_value - min_y) / max(EPS, max_y - min_y)) * (height - top - bottom)
                pts.append((x, y))
            self.draw_polyline(canvas, pts, color, 3)
        self.legend(canvas, [("chute constante", "#0f766e"), ("chute senoidal", "#b04d28")], left + 10, 22)

    def draw_solution_chart(self) -> None:
        if not self.result:
            return
        canvas = self.solution_canvas
        width, height = self.clear(canvas)
        left, top, right, bottom = self.draw_axes(canvas, width, height, "θ(t)")
        series = [
            ("chute constante", self.result["constant"], "#0f766e"),
            ("chute senoidal", self.result["sine"], "#b04d28"),
        ]
        theta_values = [v for _, sol, _ in series for v in sol.theta]
        min_y = min(theta_values)
        max_y = max(theta_values)
        pad = max(0.1, 0.15 * (max_y - min_y))
        min_y -= pad
        max_y += pad
        total = self.result["sine"].t[-1]
        for _, sol, color in series:
            pts = []
            for t_value, theta in zip(sol.t, sol.theta):
                x = left + (t_value / total) * (width - left - right)
                y = height - bottom - ((theta - min_y) / max(EPS, max_y - min_y)) * (height - top - bottom)
                pts.append((x, y))
            self.draw_polyline(canvas, pts, color, 2)
        self.legend(canvas, [(label, color) for label, _, color in series], left + 10, 22)

    def draw_linear_chart(self) -> None:
        if not self.result:
            return
        canvas = self.linear_canvas
        width, height = self.clear(canvas)
        left, top, right, bottom = self.draw_axes(canvas, width, height, "θ(t)")
        series = [
            ("não linear", self.result["sine"], "#b04d28"),
            ("linearizada: sen(θ) ≈ θ", self.result["linear"], "#355f9f"),
        ]
        theta_values = [v for _, sol, _ in series for v in sol.theta]
        min_y = min(theta_values)
        max_y = max(theta_values)
        pad = max(0.1, 0.15 * (max_y - min_y))
        min_y -= pad
        max_y += pad
        total = self.result["sine"].t[-1]
        for _, sol, color in series:
            pts = []
            for t_value, theta in zip(sol.t, sol.theta):
                x = left + (t_value / total) * (width - left - right)
                y = height - bottom - ((theta - min_y) / max(EPS, max_y - min_y)) * (height - top - bottom)
                pts.append((x, y))
            self.draw_polyline(canvas, pts, color, 2)
        self.legend(canvas, [(label, color) for label, _, color in series], left + 10, 22)

    def legend(self, canvas: tk.Canvas, items: list[tuple[str, str]], x: int, y: int) -> None:
        max_width = max((len(label) for label, _ in items), default=0) * 7 + 74
        canvas.create_rectangle(x - 8, y - 16, x + max_width, y + len(items) * 18 + 8, fill="#ffffff", outline="#d8dfd6")
        for idx, (label, color) in enumerate(items):
            yy = y + idx * 18
            canvas.create_line(x, yy, x + 22, yy, fill=color, width=3)
            canvas.create_text(x + 30, yy, text=label, anchor="w", fill="#1f2823")

    def draw_pendulum(self, canvas: tk.Canvas, theta: float, label: str, color: str, x_center: float | None = None) -> None:
        width, height = self.clear(canvas)
        cx = x_center if x_center is not None else width * 0.5
        pivot = (cx, height * 0.16)
        length = min(width * 0.34, height * 0.62)
        bob = (pivot[0] + length * math.sin(theta), pivot[1] + length * math.cos(theta))
        canvas.create_line(pivot[0], pivot[1], bob[0], bob[1], fill="#2b3630", width=4)
        canvas.create_oval(pivot[0] - 7, pivot[1] - 7, pivot[0] + 7, pivot[1] + 7, fill="#2b3630", outline="")
        radius = max(12, min(width, height) * 0.035)
        canvas.create_oval(bob[0] - radius, bob[1] - radius, bob[0] + radius, bob[1] + radius, fill=color, outline="")
        canvas.create_text(16, 18, text=label, anchor="w", fill="#1f2823")
        canvas.create_text(16, 38, text=f"θ = {theta:.4f} rad", anchor="w", fill="#667169")

    def draw_planets(self, earth_theta: float, mercury_theta: float) -> None:
        canvas = self.planet_canvas
        width, height = self.clear(canvas)
        for cx, theta, label, color in [
            (width * 0.32, earth_theta, "Terra", "#0f766e"),
            (width * 0.68, mercury_theta, "Mercúrio", "#8f5c00"),
        ]:
            pivot = (cx, height * 0.18)
            length = min(width * 0.18, height * 0.58)
            bob = (pivot[0] + length * math.sin(theta), pivot[1] + length * math.cos(theta))
            canvas.create_line(pivot[0], pivot[1], bob[0], bob[1], fill="#2b3630", width=4)
            canvas.create_oval(pivot[0] - 7, pivot[1] - 7, pivot[0] + 7, pivot[1] + 7, fill="#2b3630", outline="")
            radius = max(12, min(width, height) * 0.035)
            canvas.create_oval(bob[0] - radius, bob[1] - radius, bob[0] + radius, bob[1] + radius, fill=color, outline="")
            canvas.create_text(cx, height - 24, text=label, fill="#1f2823")

    def current_solution(self) -> Solution | None:
        if not self.result:
            return None
        key = "constant" if self.vars["choice"].get() == "chute constante" else "sine"
        return self.result[key]

    def animate(self) -> None:
        if self.result and self.mercury_result:
            if self.playing:
                self.elapsed += 0.016 * float(self.vars["speed"].get())
            sol = self.current_solution()
            if sol:
                theta = PendulumAPI.sample(sol, self.elapsed)
                self.draw_pendulum(self.pendulum_canvas, theta, self.vars["choice"].get(), "#0f766e")
            earth_theta = PendulumAPI.sample(self.result["sine"], self.elapsed)
            mercury_theta = PendulumAPI.sample(self.mercury_result["sine"], self.elapsed)
            self.draw_planets(earth_theta, mercury_theta)
        self.after(16, self.animate)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.config(text="Pausar" if self.playing else "Retomar")

    def reset_time(self) -> None:
        self.elapsed = 0.0

    def set_preset(self, kind: str) -> None:
        if kind == "earth":
            self.vars["g"].set(9.8)
        elif kind == "mercury":
            self.vars["g"].set(3.7)
        else:
            self.vars["g"].set(9.8)
            self.vars["L"].set(1.0)
            self.vars["T"].set(2.0 * math.pi)
            self.vars["alpha"].set(0.7)
            self.vars["beta"].set(0.7)
            self.vars["m"].set(120)
            self.vars["tol"].set(1e-10)
        self.recompute()


if __name__ == "__main__":
    PendulumApp().mainloop()
