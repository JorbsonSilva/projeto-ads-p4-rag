# Demonstração acadêmica do Tutor Orienta

1. **Problema.** Quem usa o Orienta precisa entender regras de mentoria, busca e agenda sem sair do aplicativo. O Tutor consulta a FAQ operacional. O assistente determinístico segue atendendo a triagem e seus fluxos próprios.
2. **Arquitetura.** Mostre o caminho Expo Web → Spring Boot → FastAPI → ChromaDB → Gemini. O Spring autentica o aluno e assina um JWT interno. O FastAPI usa uma conversa isolada em memória.
3. **Base e recuperação.** Abra [a FAQ](knowledge/orienta_faq.json) e mostre os 33 registros e suas referências. Rode o comando abaixo, após o preparo do [README](README.md). O E5 ONNX uint8 gera vetores de 384 dimensões. O ChromaDB persistente consulta dois candidatos por similaridade e aplica o limiar configurado.

   ```sh
   RAG_EMBEDDING_BACKEND=onnx python -m rag.cli --corpus orienta --max-distance 0.14 \
     --retrieve-only "Como faço para cancelar uma mentoria?"
   ```

4. **Geração e memória.** Um chatbot convencional pode responder com base em regras fixas ou no conhecimento do modelo. Com RAG, o Tutor busca trechos no corpus antes de gerar a resposta. O Gemini recebe o system prompt, os candidatos e o histórico da conversa. O identificador da evidência vem de um registro recuperado, como `M08` ou `M07`.
5. **Aplicativo.** Prepare os serviços conforme o README, abra `http://localhost:8081`, entre como aluno sintético e selecione **Dúvidas sobre o Orienta**. Faça as perguntas nesta sequência.

| Pergunta | O que mostrar |
|---|---|
| Como faço para cancelar uma mentoria? | O prazo de duas horas e a exigência de informar o motivo. Evidência `M08`. |
| E qual é o prazo? | A continuidade mantém o tema de cancelamento e recupera `M08`. Em uma conversa nova, a mesma pergunta não tem contexto suficiente. |
| Consigo trocar o horário da mentoria? | O novo horário exige antecedência mínima de 24 horas e verificação de conflitos. Evidência `M07`. |
| Como emitir passaporte? | O Tutor informa que não encontrou evidência e não exibe fontes. |

A interface mostra **Base de conhecimento** para a origem `RAG` e exibe o ID e a versão do registro. Use **Encerrar conversa** para descartar a memória. Mostre também que a triagem determinística continua disponível na tela do assistente.
