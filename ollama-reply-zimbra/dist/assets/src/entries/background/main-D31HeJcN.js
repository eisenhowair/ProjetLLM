import{b as n}from"../../../browser-polyfill-CDoadmtY.js";

const o="llama3:8b";

n.runtime.onInstalled.addListener(()=>{
	console.log("Extension installedddd")});
	n.runtime.onMessage.addListener((e,s,t)=> {
    console.log(e.tweet);
    console.log("main ton=",e.ton);

    let r = `Vous êtes un assistant IA français. Vous devez répondre au e-mails sur un ton ${e.ton}. `;

    if (e.indications) {
      r += `En suivant les indications.
      Indications :
      ${e.indications}`;
    }

    console.log(r);

		if(e.action==="fetchResponse")
			return i(e.tweet, r).then(a=>{
				t({response:a})
			}),!0
		});

async function i(e, r) {
  try {
    const response = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream: false, model: o, messages: [{ role: "system", content: r }, { role: "user", content: e }] })
    });
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const data = await response.json();
    console.log(data.message.content);
    return data.message.content;
  } catch (error) {
    console.error("Error fetching response:", error);
    return "Désolé, une erreur s'est produite. Veuillez réessayer plus tard.";
  }
}
