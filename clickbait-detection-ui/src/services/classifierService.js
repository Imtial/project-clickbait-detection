import http from "./httpService";

export async function getPredictions(headlineOrUrl) {
  const isUrl =
    headlineOrUrl.indexOf("http://") === 0 ||
    headlineOrUrl.indexOf("https://") === 0;
  const { data } = isUrl
    ? await http.get("url=" + headlineOrUrl)
    : await http.get(headlineOrUrl);
  return data;
}

export async function sendFeedback(feedback) {
  const { data } = http.post("/", feedback);
  return data;
}
