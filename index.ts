import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Elysia, t } from "elysia";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { retriever } from "./utils/retriever";
import type { Document } from "langchain";
import { combineDocuments } from "./utils/misc";

const supabaseUrl = process.env.SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_KEY!;
const openAIApiKey = process.env.OPENAI_API_KEY!;

const app = new Elysia();

async function syncDocuments() {
  const file = Bun.file("./docs/system-prompt-success.md");
  const text = await file.text();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    separators: ["\n\n", "\n", " ", ""],
    chunkOverlap: 50,
  });
  const output = await splitter.createDocuments([text]);

  const client = createClient(supabaseUrl, supabaseKey);

  await SupabaseVectorStore.fromDocuments(
    output,
    new OpenAIEmbeddings({ openAIApiKey, model: "text-embedding-3-small" }),
    {
      client,
      tableName: "documents",
    },
  );
}

async function handlePrompt(userInput: string) {
  const llm = new ChatOpenAI({
    openAIApiKey,
    model: "gpt-5-nano",
  });
  const standaloneQuestionTemplate = `Given a question, convert it to a standalone question. question: {question} standalone question:`;
  const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
    standaloneQuestionTemplate,
  );
  const answerTemplate = `
You are a helpful and enthusiastic support bot who can answer a given question about professional experience on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I', sorry, I don't know the answer to that" and direct the questioner to email me@aaliyev.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer:
`.trim();

  const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

  const standaloneChain = standaloneQuestionPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());

  const retrieverChain = RunnableSequence.from([
    ({ standalone_question }) => standalone_question,
    retriever,
    combineDocuments,
  ]);
  const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

  const chain = RunnableSequence.from([
    {
      standalone_question: standaloneChain,
      original_input: new RunnablePassthrough(),
    },
    {
      context: retrieverChain,
      question: ({ original_input }) => original_input.question,
    },
    answerChain,
  ]);

  const response = await chain.invoke({
    question: userInput,
  });

  console.log("response = ", response);
  return response;
}

app.get("/", () => "Hello");
app.post(
  "/prompt",
  async ({ body, set }) => {
    try {
      console.log("prompt = ", body.prompt);
      const res = await handlePrompt(body.prompt);
      return res;
    } catch (error) {
      set.status = 500;
      console.error(error);
      return { error };
    }
  },
  {
    body: t.Object({ prompt: t.String() }),
  },
);

async function startServer() {
  try {
    // await syncDocuments();
    app.listen(3333, () => console.log("Running on port 3333"));
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}

startServer();

//   const prompt = PromptTemplate.fromTemplate(template);
//   const toStandAlonePrompt = PromptTemplate.fromTemplate(userInput);
//   const chain = prompt.pipe(llm);
//   const response = await chain.invoke({ something: "something" });
//   const content = response.content;
