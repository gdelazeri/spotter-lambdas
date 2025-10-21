// lambda/imageProcessorBedrock.js
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { RekognitionClient, DetectLabelsCommand } from "@aws-sdk/client-rekognition";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { MongoClient, ObjectId } from "mongodb";
import { QdrantClient } from "@qdrant/js-client-rest";
import { v4 as uuidv4 } from "uuid";

const s3 = new S3Client({ region: process.env.AWS_REGION });
const rekognition = new RekognitionClient({ region: process.env.AWS_REGION });
const bedrock = new BedrockRuntimeClient({ region: "us-east-1" }); // Bedrock disponível aqui
const mongo = new MongoClient(process.env.MONGO_URI);
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL, apiKey: process.env.QDRANT_API_KEY });

// Utilitário para converter stream em buffer
function streamToBufferPromise(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on("data", (c) => chunks.push(c));
    stream.on("end", () => resolve(Buffer.concat(chunks)));
    stream.on("error", reject);
  });
}

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

      // 1️⃣ Baixa imagem do S3
      const imageBuffer = await downloadImage(bucket, key);

      // 2️⃣ Usa Rekognition para gerar descrição da imagem
      const rekResp = await rekognition.send(
        new DetectLabelsCommand({
          Image: { Bytes: imageBuffer },
          MaxLabels: 10,
          MinConfidence: 75,
        })
      );

      // Monta uma descrição textual simples baseada nas labels
      const labels = rekResp.Labels.map((l) => l.Name).join(", ");
      const imageDescription = `A imagem contém: ${labels.toLowerCase()}.`;

      console.log("Descrição da imagem:", imageDescription);

      // 3️⃣ Combina legenda + descrição da imagem
      const combinedText = caption
        ? `${caption} ${imageDescription}`.trim()
        : imageDescription;

      // 4️⃣ Gera embeddings com Titan Embeddings V2
      const body = JSON.stringify({
        inputText: combinedText,
      });

      const embedResp = await bedrock.send(
        new InvokeModelCommand({
          modelId: "amazon.titan-embed-text-v2:0",
          contentType: "application/json",
          accept: "application/json",
          body,
        })
      );

      const responseBody = JSON.parse(Buffer.from(await embedResp.body).toString());
      const embedding = responseBody.embedding;

      const embeddingId = uuidv4();

      // 5️⃣ Atualiza MongoDB
      await posts.updateOne(
        { _id: new ObjectId(postId) },
        {
          $set: {
            status: "processed",
            embeddingId,
          },
        }
      );

      // 6️⃣ Upsert no Qdrant
      await qdrant.upsert("posts", {
        points: [
          {
            id: embeddingId,
            vector: embedding,
            payload: { postId, userId, caption, imageDescription },
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
