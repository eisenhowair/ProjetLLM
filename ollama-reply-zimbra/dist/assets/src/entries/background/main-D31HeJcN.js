import{b as n}from"../../../browser-polyfill-CDoadmtY.js";

const o="llama3:8b";

n.runtime.onInstalled.addListener(()=>{
	console.log("Extension installed")});
	n.runtime.onMessage.addListener((e,s,t)=> {
    console.log(e.tweet);

	let r = `
 
 	Vous êtes un assistant IA français spécialisé dans la gestion des e-mails. Répondez aux e-mails en utilisant un ton ${e.ton}. Veuillez suivre les instructions suivantes :
	1. Respectez les indications fournies : ${e.indications}.
	2. Adaptez votre langage et votre style en fonction du destinataire de l'e-mail.
	3. Assurez-vous que toutes les réponses soient polies, professionnelles et respectueuses.
	4. Relisez votre réponse pour vérifier la grammaire et l'orthographe avant de l'envoyer.
	5. Fournissez des informations supplémentaires si nécessaire pour clarifier ou enrichir la réponse.
	`

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
