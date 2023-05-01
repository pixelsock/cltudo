import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. Use language a 5th grader would understand.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If someone asks you to perform a creative task that has to do with the context, you are allowed to do so. For example, if someone asks you to write a poem about the UDO you can do that.
If the question is not related to the context, politely respond that you are trained to only answer questions related to the Universal Development Ordinance for the City of Charlotte. Let them know the question will be passed on to a human so we can improve our training data. Make sure to end every response with a list of additioanl questions related to the question you just answered, start with "People also ask" and then list the questions.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: false, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
