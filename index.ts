import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Elysia, t } from "elysia";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough } from "@langchain/core/runnables";

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
  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
  const client = createClient(supabaseUrl, supabaseKey);
  const vectorStore = new SupabaseVectorStore(embeddings, {
    client,
    tableName: "documents",
    queryName: "match_documents",
  });
  const retriever = vectorStore.asRetriever();

  const llm = new ChatOpenAI({
    openAIApiKey,
    model: "gpt-5-nano",
  });
  const standaloneQuestionTemplate = `Given a question, convert it to a standalone question. question: {question} standalone question:`;
  const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
    standaloneQuestionTemplate,
  );
  const chain = standaloneQuestionPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .pipe(
      new RunnablePassthrough({
        func: (input: string) => {
          console.log("Standalone Question:", input);
        },
      }),
    )
    .pipe(retriever);
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
