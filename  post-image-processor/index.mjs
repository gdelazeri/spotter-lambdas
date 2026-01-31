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

const POST_STATUS = {
  PENDING: "pending",
  PROCESSED: "processed",
  CENSORED: "censored",
  FAILED: "failed",
}

async function processPost(record) {
  const posts = mongo.db("spotter").collection("posts");

  let postId;

  try {
    const bucket = record.s3.bucket.name;
    const key = decodeURIComponent(record.s3.object.key.replace(/\+/g, " "));
    postId = key.split("/")[2].split(".")[0];

    const post = await posts.findOne({ _id: new ObjectId(postId) });
    if (!post) {
      console.warn(`Post ${postId} not found.`);
      return;
    }

    const { userId, caption = "" } = post;

    // 1️⃣ Baixa imagem
    const imageBuffer = await downloadImage(bucket, key);
    const base64Image = imageBuffer.toString("base64");

    // 2️⃣ Descrição da imagem (multimodal)
    const visionResponse = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "Você descreve imagens de forma objetiva e identifica possíveis conteúdos sensíveis.",
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: caption
                ? `Descreva a imagem considerando esta legenda: "${caption}". Aponte se há nudez, violência ou conteúdo sexual.`
                : "Descreva a imagem e aponte se há nudez, violência ou conteúdo sexual.",
            },
            {
              type: "image_url",
              image_url: { url: `data:image/jpeg;base64,${base64Image}` },
            },
          ],
        },
      ],
    });

    const imageDescription = visionResponse.choices[0].message.content.trim();
    console.log("Descrição da imagem:", imageDescription);

    // 3️⃣ 🔥 MODERATION API (OFICIAL)
    const moderationInput = caption
      ? `${caption} ${imageDescription}`.trim()
      : imageDescription;

    const moderationResp = await openai.moderations.create({
      model: "omni-moderation-latest",
      input: moderationInput,
    });

    const moderationResult = moderationResp.results[0];

    if (moderationResult.flagged) {
      console.warn("Conteúdo sensível detectado:", moderationResult.categories);

      await posts.updateOne(
        { _id: new ObjectId(postId) },
        {
          $set: {
            status: POST_STATUS.CENSORED,
            processedAt: Date.now(),
            moderation: {
              flagged: true,
              categories: moderationResult.categories,
            },
            imageDescription,
          },
        }
      );

      return;
    }

    // 4️⃣ Embeddings
    const embeddingResp = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: moderationInput,
    });

    const embedding = embeddingResp.data[0].embedding;

    // 5️⃣ Atualiza Mongo
    await posts.updateOne(
      { _id: new ObjectId(postId) },
      {
        $set: {
          status: POST_STATUS.PROCESSED,
          processedAt: Date.now(),
          imageDescription,
        },
      }
    );

    // 6️⃣ Qdrant
    await qdrant.upsert("posts", {
      points: [
        {
          id: uuidv4(),
          vector: embedding,
          payload: {
            postId,
            userId,
            caption,
            imageDescription,
            createdAt: new Date(),
          },
        },
      ],
    });

    console.log(`Post ${postId} processed successfully.`);
  } catch (error) {
    console.error("Error on process post:", error);

    if (postId) {
      await posts.updateOne(
        { _id: new ObjectId(postId) },
        { $set: { status: POST_STATUS.FAILED, error: error.message } }
      );
    }
  }
}

export const handler = async (event) => {
  console.log("Event received:", JSON.stringify(event, null, 2));

  try {
    await mongo.connect();

    for (const record of event.Records) {
      await processPost(record);
    }

    return { statusCode: 200, body: JSON.stringify({ success: true }) };
  } catch (error) {
    console.error("Error on process posts:", error);
    return { statusCode: 500, body: JSON.stringify({ error: error.message }) };
  } finally {
    await mongo.close();
  }
};
