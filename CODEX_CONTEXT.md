# Contexto do Projeto para Codex

Este arquivo serve como memoria operacional do projeto para quando ele for aberto em outra maquina ou em uma nova sessao do Codex.

## Objetivo

O projeto implementa um software em Python/Tkinter para validacao/comparacao de modelos eletricos a partir de sinais experimentais e sinais simulados/modelados.

O usuario conversa em portugues, mas a interface do software deve ficar em ingles.

O software deve permitir:

- Analise de teste individual.
- Validacao completa de um modelo com varios test cases e varios runs.
- Importacao de sinais senoidais ou grandezas escalares via CSV.
- Sincronizacao entre experimental e modelo.
- Separacao automatica e manual de janelas de teste.
- Calculo de metricas, scores, graficos radar e relatorio PDF.
- Evolucao para sinais monofasicos e trifasicos dentro de uma unica versao do app.

## Arquivos Principais

- `main.py`: interface Tkinter, fluxo do app, metricas, tabelas, scores, graficos, relatorio PDF e fluxo de validacao completa.
- `app/signal_processing.py`: leitura/parse de CSV, padronizacao de amostragem, sincronizacao, calculo de escalares, janelas de teste e processamento de sinais.
- `app/project_models.py`: modelo de dados do projeto de validacao completa.
- `app/project_storage.py`: salvamento/carregamento dos projetos de validacao em JSON.
- `build_exe.bat`: build do executavel via PyInstaller.
- `comparador_modelos.spec`: configuracao do PyInstaller.

## Como Rodar

Na pasta do projeto:

```powershell
.\.venv\Scripts\activate
python main.py
```

Para gerar executavel:

```powershell
.\build_exe.bat
```

Historicamente, o Google Drive causou problemas de permissao com `.git`, `dist` e algumas pastas de build. O ideal e manter o repositorio em uma pasta local fora do Drive, por exemplo `C:\Projetos\API_validacao_modelo`.

## Estado Atual Funcional

O fluxo monofasico estava funcionando e deve ser preservado.

Fluxo individual:

- Tela inicial com escolha entre `Single test analysis` e `Complete model validation`.
- Testes disponiveis:
  - `Steady-state tests`
  - `Step test`
  - `Transient disturbance test`
  - `Ramp test`
- Importacao de sinais:
  - `Import sinusoidal signals`
  - `Import scalar signals`
- Classificacao das colunas CSV via combobox.
- Sincronizacao.
- Plots em:
  - valores reais
  - PU independente
  - PU usando base experimental
- Usuario escolhe a representacao usada nos calculos.
- Separacao de janelas com tolerancia ajustavel.
- Possibilidade de ajuste manual das fronteiras de janela nos plots.
- Selecao de metricas e thresholds.
- Calculo de tabelas, scores e radar charts.

Fluxo de validacao completa:

- Criacao/carregamento de `Model validation`.
- Definicao de test cases, numero de capturas experimentais/modelo e condicoes.
- Upload dos arquivos.
- Classificacao compartilhada de colunas quando os nomes coincidem.
- Configuracoes globais de metricas, filtros, representacao, tolerancias, normalizacao, scores etc.
- Runs por padrao em `all x all`.
- Tela final com abas:
  - `Executive summary`
  - `By test type`
  - `By quantity`
  - `By category`
  - `Test cases`
  - `Radar charts`
- Botao para salvar resultados.
- Botao para gerar relatorio PDF.
- Projetos salvos podem ser carregados e devem abrir diretamente nos resultados quando ja houver runs processados.

## Decisoes Importantes

- Manter uma unica versao do software, com suporte monofasico e trifasico no mesmo app.
- Nao quebrar o fluxo monofasico existente.
- Interface sempre em ingles.
- Conversa e explicacoes ao usuario em portugues.
- Nomes internos como `fault_test`, `pre_fault_percent` podem permanecer por compatibilidade, mas textos visiveis devem usar `disturbance`, nao `fault`.
- O indice antes chamado de `Phase error` foi renomeado visualmente para `Phase delay`.
- Metricas senoidais devem continuar sendo calculadas ciclo a ciclo, descartando ciclos incompletos, mas apenas em ciclos dentro de janelas nao transitorias.
- Para sinais senoidais, janelas transitorias nao entram nas metricas de forma de onda.
- Para projetos salvos, o app deve buscar tanto em `storage/assessment_projects` quanto no fallback temporario `%TEMP%\ModelValidationProjects`.

## Suporte Trifasico Implementado Inicialmente

Foi adicionada uma primeira versao do suporte trifasico, ainda a ser refinada.

Configuracao adicionada:

- `Electrical system`
  - `Single-phase`
  - `Three-phase, 3-wire`
  - `Three-phase, 4-wire`

Opcoes de importacao senoidal adicionadas:

- `Voltage A`
- `Voltage B`
- `Voltage C`
- `Current A`
- `Current B`
- `Current C`

Opcoes de importacao escalar adicionadas:

- `RMS voltage A or DC`
- `RMS voltage B or DC`
- `RMS voltage C or DC`
- `RMS current A or DC`
- `RMS current B or DC`
- `RMS current C or DC`
- `Active power A/B/C`
- `Reactive power A/B/C`
- `Zero-sequence voltage`
- `Positive-sequence voltage`
- `Negative-sequence voltage`
- `Zero-sequence current`
- `Positive-sequence current`
- `Negative-sequence current`
- `Voltage zero-sequence unbalance`
- `Voltage negative-sequence unbalance`
- `Current zero-sequence unbalance`
- `Current negative-sequence unbalance`

Para sinais senoidais trifasicos, `calculate_scalar_signals` calcula ciclo a ciclo:

- Tensao RMS por fase.
- Corrente RMS por fase.
- Frequencia.
- Potencia ativa por fase.
- Potencia reativa por fase.
- Potencia ativa total.
- Potencia reativa total.
- Componentes de sequencia zero, positiva e negativa para tensao e corrente.
- Desequilibrio de sequencia zero/negativa relativo a sequencia positiva.

Para sinais escalares trifasicos, `augment_scalar_dataset` deriva automaticamente quando possivel:

- `voltage` como media de `voltage_a/b/c`.
- `current` como media de `current_a/b/c`.
- `active_power` como soma de `active_power_a/b/c`.
- `reactive_power` como soma de `reactive_power_a/b/c`.

As novas grandezas foram adicionadas a `SCALAR_COLUMN_ORDER` e `SCALAR_DISPLAY_LABELS`, para reaproveitar plots, janelas, metricas escalares, scores, radar charts e relatorio.

## Metricas e Scores

Categorias atuais de score:

- `Waveform fidelity`
- `Spectral and harmonic fidelity`
- `Steady-state operating point accuracy`
- `Steady-state oscillation and variability`
- `Transient magnitude accuracy`
- `Transient timing accuracy`
- `Shape transient similarity - extra indices`

Scores sao de 0 a 100.

Ha variantes:

- `Original`
- `Phase-delay-corrected` quando habilitado para metricas senoidais.
- `Delay-adjusted` quando habilitado para metricas transitorias.

Radar charts:

- Gerados por grandeza e por categoria.
- Removem eixos sem valores.
- Se sobrarem apenas dois eixos, duplicam alternadamente para evitar radar degenerado em reta.

## Cuidados Tecnicos

- Nao mexer no fluxo monofasico sem testar.
- Evitar alterar nomes internos salvos em JSON quando eles ja existem em projetos antigos.
- Usar `apply_patch` para edicoes manuais.
- Evitar versionar:
  - `.venv`
  - `build`
  - `dist`
  - `dist_exe`
  - `build_exe_tmp`
  - `storage`
  - `__pycache__`
- O arquivo `.gitignore` ja foi criado com esses ignores.
- Se o repo estiver em Google Drive, mover/clonar para uma pasta local fora do Drive.

## Validacoes Usuais

Depois de alteracoes, rodar:

```powershell
@'
import ast
from pathlib import Path
for path in ('main.py', 'app/signal_processing.py', 'app/project_models.py', 'app/project_storage.py', 'app/__init__.py'):
    ast.parse(Path(path).read_text(encoding='utf-8'))
    print(path, 'ok')
'@ | python -
```

E:

```powershell
@'
import main
import app.signal_processing
import app.project_storage
print('import ok')
'@ | python -
```

Para testar rapidamente o calculo trifasico, pode criar sinais senoidais sinteticos equilibrados e chamar `calculate_scalar_signals`.

## Proximos Passos Sugeridos

- Revisar visualmente o fluxo monofasico depois da adicao trifasica.
- Testar um CSV trifasico real em modo senoidal.
- Testar um CSV trifasico real em modo escalar.
- Refinar sincronismo trifasico:
  - decidir se sincroniza por fase A, grandeza agregada ou sequencia positiva.
- Refinar metricas especificas de sequencia:
  - erro de amplitude V1/I1
  - erro de fase V1/I1
  - erro V2/V1
  - erro I2/I1
  - erro V0/V1 e I0/I1, especialmente para sistema 4 fios
- Avaliar se `Three-phase, 3-wire` deve esconder ou tratar de forma diferente sequencia zero.
- Melhorar relatorio para agrupar fases e sequencias de forma mais legivel.
- Criar testes automatizados pequenos para:
  - parse CSV
  - calculo monofasico
  - calculo trifasico equilibrado
  - componentes de sequencia

## Observacao para Nova Sessao do Codex

Ao abrir este projeto em uma nova sessao, leia este arquivo antes de editar. O usuario quer evoluir o suporte trifasico aos poucos, mantendo tudo em uma versao unica e preservando o monofasico que ja funciona.
