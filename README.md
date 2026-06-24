# API Validacao de Modelo

Projeto em Python com:

- API FastAPI para futuras integracoes.
- Aplicativo desktop para importar os arquivos de comparacao.

## Executar o aplicativo desktop

No Windows, execute:

```bat
run_app.bat
```

Funcionalidade atual:

- Tela inicial para selecionar o teste comparativo.
- Tela de parametros do teste escolhido antes da importacao.
- Tela para escolher entre importar sinais senoidais ou sinais escalares.
- Tela de upload dos sinais senoidais com um `.csv` para sinais experimentais e um `.csv` para sinais do modelo.
- Tela de upload dos sinais escalares com um `.csv` para sinais experimentais e um `.csv` para sinais do modelo.
- Mapeamento dinamico das colunas do CSV com opcoes de classificacao para senoidais e escalares.
- Botao para continuar a validacao apos importar e classificar todas as colunas.
- Padronizacao de amostragem, calculo de grandezas escalares para sinais senoidais e sincronismo automatico.
- Tela de plots com comparacao entre experimental e modelo, incluindo zoom do sincronismo.

## Executar a API

No Windows, execute:

```bat
run_api.bat
```

A API sobe por padrao em:

```text
http://127.0.0.1:8000
```

Documentacao automatica:

```text
http://127.0.0.1:8000/docs
```

## Gerar o executavel

No Windows, execute:

```bat
build_exe.bat
```

O executavel sera gerado em:

```text
dist\comparador_modelos.exe
```
