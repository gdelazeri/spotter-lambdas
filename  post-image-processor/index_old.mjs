// lambda/imageProcessor.js
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { MongoClient, ObjectId } from "mongodb";
import OpenAI from "openai";
import { QdrantClient } from "@qdrant/js-client-rest";
import { v4 as uuidv4 } from "uuid";

const s3 = new S3Client({ region: process.env.AWS_REGION });
const mongo = new MongoClient(process.env.MONGO_URI);
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL, apiKey: process.env.QDRANT_API_KEY });

// Utilitário: converte stream em buffer
function streamToBufferPromise(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on("data", (c) => chunks.push(c));
    stream.on("end", () => resolve(Buffer.concat(chunks)));
    stream.on("error", reject);
  });
}

// Baixa imagem do S3
async function downloadImage(bucket, key) {
  const resp = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  return streamToBufferPromise(resp.Body);
}

export const handler = async (event) => {
  console.log("event", JSON.stringify(event, null, 2));

  try {
    await mongo.connect();
    const posts = mongo.db("spotter").collection("posts");

    for (const record of event.Records) {
      const bucket = record.s3.bucket.name;
      const key = decodeURIComponent(record.s3.object.key.replace(/\+/g, " "));
      const postId = key.split("/")[2].split(".")[0]; // posts/<userId>/<postId>.jpg

      const post = await posts.findOne({ _id: new ObjectId(postId) });
      if (!post) {
        console.warn(`Post ${postId} não encontrado.`);
        continue;
      }

      const { userId, caption = "" } = post;

      // 1️⃣ Baixa imagem
      const imageBuffer = await downloadImage(bucket, key);
      const base64Image = imageBuffer.toString("base64");

      // 2️⃣ Usa GPT-4o-mini (multimodal) para gerar descrição da imagem
      const visionResponse = await openai.chat.completions.create({
        model: "gpt-4o-mini", // modelo multimodal leve e barato
        messages: [
          {
            role: "system",
            content: "Você é um assistente que descreve imagens de forma objetiva e detalhada.",
          },
          {
            role: "user",
            content: [
              { type: "text", text: "Descreva brevemente o conteúdo e o contexto desta imagem:" },
              { type: "image_url", image_url: { url: `data:image/jpeg;base64,${base64Image}` } },
            ],
          },
        ],
      });

      const imageDescription = visionResponse.choices[0].message.content.trim();
      console.log("Descrição da imagem:", imageDescription);

      // 3️⃣ Combina legenda + descrição da imagem
      const combinedText = caption
        ? `${caption} ${imageDescription}`.trim()
        : imageDescription;

      // 4️⃣ Gera embeddings a partir do texto combinado
      const embeddingResp = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: combinedText,
      });
      const embedding = embeddingResp.data[0].embedding;

      // 5️⃣ Atualiza MongoDB
      await posts.updateOne(
        { _id: new ObjectId(postId) },
        {
          $set: {
            status: "processed",
            imageDescription
          },
        }
      );

      // 6️⃣ Upsert no Qdrant
      await qdrant.upsert("posts", {
        points: [
          {
            id: uuidv4(),
            vector: embedding,
            payload: { postId, userId, caption, imageDescription, createdAt: new Date() },
          },
        ],
      });

      console.log(`✅ Post ${postId} processado com sucesso.`);
    }

    return { statusCode: 200, body: JSON.stringify({ success: true }) };
  } catch (error) {
    console.error("Erro ao processar imagem:", error);
    return { statusCode: 500, body: JSON.stringify({ error: error.message }) };
  } finally {
    await mongo.close();
  }
};
