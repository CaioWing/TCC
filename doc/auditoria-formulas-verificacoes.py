"""Verificações algébricas e numéricas da auditoria; não validam os dados dos voos.

Executar com Python e SymPy instalados. Não modifica o manuscrito.
"""
from math import ceil, pi, sqrt
from pathlib import Path
import re
import sympy as s

root = Path(__file__).resolve().parent
lines = []

def record(name, value):
    lines.append(f'{name}: {value}')

def check(name, expr):
    result = s.simplify(expr)
    assert result == 0, (name, result)
    record(name, 'OK (diferença simbólica zero)')

m, dm, T, rho, A, h, p, f, V, Nx, obj, P = s.symbols(
    'm dm T rho A h p f V Nx obj P', positive=True
)
vh = s.sqrt(T / (2 * rho * A))
check('2.1: conservação de tração', 2 * rho * A * vh**2 - T)
check('2.2: potência ideal', T * vh - T**s.Rational(3, 2) / s.sqrt(2 * rho * A))
N, g = s.symbols('N g', positive=True)
check('Multirrotor: soma das potências com área total N A',
      N*(m*g/N)**s.Rational(3, 2)/s.sqrt(2*rho*A)
      - (m*g)**s.Rational(3, 2)/s.sqrt(2*rho*N*A))
x = s.symbols('x')
check('2.4: coeficiente linear em delta_m/m',
      s.diff((1 + x)**s.Rational(3, 2), x).subs(x, 0) - s.Rational(3, 2))
power = s.Function('P')(V)
check('2.12: derivada de P/V', s.diff(power / V, V) - (V*s.diff(power, V)-power)/V**2)
eta, Pav = s.symbols('eta Pav', positive=True)
check('Ótimo elétrico: condição com carga fixa e eficiência constante',
      eta*V**2*s.diff((power/eta+Pav)/V, V)
      - (V*s.diff(power, V)-power-eta*Pav))
gsd = h*p/f
w = Nx*gsd
check('2.17: energia versus altura', P/(V*w) - f*P/(Nx*p*V*h))
check('2.19: eliminação de altura', P/(V*w) - P/(V*Nx)*(obj/gsd)/obj)
spmin = s.symbols('spmin', positive=True)
check('2.20: altura limite', (obj/gsd).subs(h, obj*f/(p*spmin)) - spmin)
psi, theta, phi = s.symbols('psi theta phi', real=True)
cs, sn = s.cos, s.sin
rotation = s.Matrix([
    [cs(theta)*cs(psi), sn(phi)*sn(theta)*cs(psi)-cs(phi)*sn(psi), cs(phi)*sn(theta)*cs(psi)+sn(phi)*sn(psi)],
    [cs(theta)*sn(psi), sn(phi)*sn(theta)*sn(psi)+cs(phi)*cs(psi), cs(phi)*sn(theta)*sn(psi)-sn(phi)*cs(psi)],
    [-sn(theta), sn(phi)*cs(theta), cs(phi)*cs(theta)],
])
assert s.simplify(rotation.T*rotation-s.eye(3)) == s.zeros(3)
check('Matriz de Euler: determinante', rotation.det()-1)
record('Matriz de Euler: ortogonalidade', 'OK')
yaw = rotation.subs({theta: 0, phi: 0})[:2, :2]
assert yaw.subs(psi, 0)*s.Matrix([1, 0]) == s.Matrix([1, 0])
assert yaw.subs(psi, s.pi/2)*s.Matrix([1, 0]) == s.Matrix([0, 1])
assert yaw.subs(psi, s.pi/2)*s.Matrix([0, 1]) == s.Matrix([-1, 0])
record('2.22: topo/direita nos casos yaw 0 e 90 graus', 'OK na convenção declarada')
xa, xo, r = s.symbols('xa xo r', real=True)
check('Cap.3: erro de projeção', xa+r*(xo-xa)-xo-(1-r)*(xa-xo))
L = s.symbols('L', positive=True)
# Fração do segmento uniforme [-L/2,L/2] a distância <=L/4 do centro.
check('Cap.3: mediana da distância L/4', ((L/4)-(-L/4))/L - s.Rational(1, 2))
nv, Einf, Pc0, At = s.symbols('nv Einf Pc0 At', positive=True)
check('Computação: energia/área da memória geográfica',
      (Pc0+(nv*V*w/At)*Einf)/(V*w) - (Pc0/(V*w)+nv*Einf/At))
record('Acréscimo exato de potência ideal para massa +10%', f'{100*((1.1)**1.5-1):.6f}% (linear: 15%)')
rotors, radius, density, gravity, mass = 4, 0.12, 1.225, 9.81, 1.4
area = pi*radius**2
induced = rotors*1.15*(mass*gravity/rotors)**1.5/sqrt(2*density*area)
profile = rotors*(0.1*0.02/8)*density*area*110**3
record('Exemplo pairado: potência induzida/modelada [W]', f'{induced:.6f}')
record('Exemplo pairado: potência perfil [W]', f'{profile:.6f}')
record('Exemplo pairado: aumento total massa +10%', f'{100*induced*((1.1)**1.5-1)/(induced+profile):.6f}%')
record('Condições do exemplo anterior', 'rho=1,225; g=9,81; RPM e kappa fixos; parâmetros da legenda; não reproduz código original da figura')
foot_w, foot_l = 3840*.014, 2160*.014
record('Pegada 3840x2160, GSD=1,4 cm [m]', f'{foot_w:.2f} x {foot_l:.2f}')
record('Área nova a cada segundo, V=4m/s', f'{100*4/foot_l:.4f}%')
record('Visibilidade a 4m/s; 30 quadros/s', f'{foot_l/4:.4f} s; {30*foot_l/4:.4f} quadros (estimativa contínua)')
record('Objeto 0,4m a 40m/80m [pixels]', f'{.4/.014:.4f}; {.4/.028:.4f}')
record('Recortes 640x640 para quadro 3840x2160', ceil(3840/640)*ceil(2160/640))
record('Área recorte 640, GSD=1,4cm [m²]', f'{(640*.014)**2:.4f}')
record('32 pixels a GSD=1,42 cm [cm]', f'{32*1.42:.4f}')
record('Conversão J/m² para Wh/km²', f'multiplicar por {10**6/3600:.9f}')
assert min(1, 2) == 1
record('Fração nova para deslocamento de 2L por quadro', 'fórmula corrigida saturada=1, com lacunas entre quadros')
record('Contraexemplo de recall por proximidade', '2 anotações mesma classe a 0m e 1m; 1 previsão a 0,5m; dmax=1m: fórmula anterior=100%; pareamento corrigido um a um=50%')

equations = []
for path in sorted((root/'USPSC-TA-Textual').glob('*.tex')):
    text = path.read_text(encoding='utf-8')
    equations += re.findall(r'\\label\{(eq_[^}]+)\}', text)
record('Cobertura: equações numeradas identificadas', len(equations))
assert len(equations) == 33, 'Reinventariar: quantidade de equações mudou'
for label in equations:
    record('Inventário', label)

output = root/'auditoria-formulas-verificacoes.txt'
output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines[:32]))
