import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_KEY!;
const openAIApiKey = process.env.OPENAI_API_KEY!;

const embeddings = new OpenAIEmbeddings({
  openAIApiKey,
  model: "text-embedding-3-small",
});
const client = createClient(supabaseUrl, supabaseKey);

const vectorStore = new SupabaseVectorStore(embeddings, {
  client,
  tableName: "documents",
  queryName: "match_documents",
});

const retriever = vectorStore.asRetriever();

export { retriever };
