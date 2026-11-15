# Changelog

## 1.0.1
- Renamed from "Shield·Code" to **Shield Code**, with a plain space. The ID `aaron43210.shield-code` is unchanged, so installed copies update normally.

## 1.0.0
First stable release. One version number now covers the extension, its bundled engine, the web app and the API.
- **Shield: Run AI Agent Swarm on Workspace (MCP)** now appears in the Command Palette.
- Windows: saving a file no longer leaves its old findings in the list.
- SARIF export reports a failed scan instead of saying the file was written.
- The agent swarm log shows the notes agents send each other.
- Default AI models come from the engine, so a retired model name cannot linger in the extension.

## 0.5.2
- Fix README images on the store pages.
- License: AGPL-3.0.

## 0.5.1
- Release builds now compile the extension. First Open VSX release.

## 0.5.0
- **Shield: AI Deep Scan**: asks, then sends image, document and text-note content to the connected provider to find sensitive data no rule covers.
- **Why flagged** and **Fix…** on findings in the tree, so image and document findings can be fixed.
- Why flagged and Explain now appear in the lightbulb menu.
- OCR: keys split by a space or holding a misread glyph are still found.

## 0.2.0
- Workspace scan with parallel analyzers, diagnostics, findings tree, fix preview + confirm, AI Provider panel.
