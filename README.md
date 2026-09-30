# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

### Installation

```
$ npm ci
```

### Local Development

```
$ npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

#### Vercel

Vercel uses `npm ci` and `npm run build`, then serves `build/`, as configured in
`vercel.json`. Commit both `package.json` and `package-lock.json` when changing
dependencies. The project uses the public npm registry through `.npmrc`.

`vscode-languageserver-types` is an explicit runtime dependency because Mermaid's
Langium parser imports it directly without declaring it in its own dependencies.
Keep it installed even though this site does not use a VS Code language server.

To verify a deployment from a clean install, run `npm ci` followed by
`npm run build`. When retrying a previously failed Vercel deployment after this
dependency fix, redeploy with **Use existing Build Cache** unchecked.

#### GitHub Pages

Using SSH:

```
$ USE_SSH=true npm run deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
# learn-ai-ml
