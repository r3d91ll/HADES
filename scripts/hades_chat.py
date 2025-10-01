#!/usr/bin/env python3
"""
Module: scripts/hades_chat.py
Summary: Interactive HADES agent chat with experiential memory and PathRAG retrieval.
Owners: @todd, @hades-runtime
Last-Updated: 2025-09-30
Inputs: ARANGO_* env (sockets/auth), QWEN_* model envs; CLI flags
Outputs: Conversational turns persisted to Arango via store; optional retrieval
Data-Contracts: messages/core_memory collections (via ArangoMemoryStore)
Related: core/runtime/memory/*, Docs/Program_Plan.md (Phases E)
Stability: beta; Security: RO for retrieval, RW only for chat storage
Boundary: C_ext limited by persona/human templates; P_ij requires valid session id

Usage:
  poetry run python scripts/hades_chat.py --session demo --dry-run
  poetry run python scripts/hades_chat.py --session demo

Environment:
  - ARANGO_DB_NAME, ARANGO_RO_SOCKET, ARANGO_RW_SOCKET, ARANGO_PASSWORD
  - QWEN_* (see .env.example)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    resolve_memory_config,
)
from core.embedders.embedders_jina import JinaV4Embedder
from core.runtime.memory import (
    ConsoleTelemetryClient,
    MemGPTOrchestrator,
    OrchestratorConfig,
    QwenModelConfig,
)
from core.runtime.memory.memory_store import ArangoMemoryStore
from core.runtime.memory.pathrag_code import CodePathRAG, PathRAGCaps
from core.runtime.memory.session import MemGPTSession

# GraphSAGE integration (optional)
try:
    from core.gnn.inference import GraphSAGEInference
    from core.gnn.graph_builder import GraphBuilder
    GRAPHSAGE_AVAILABLE = True
except ImportError:
    GRAPHSAGE_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an interactive MemGPT chat session")
    p.add_argument("--session", default="", help="Session ID (random UUID if omitted)")
    p.add_argument("--restore", type=int, default=50, help="Restore up to N prior turns")
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit without loading model")
    p.add_argument("--no-schema", action="store_true", help="Skip ensure_schema (use when DB is already provisioned)")
    p.add_argument("--telemetry", action="store_true", help="Print telemetry spans")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    # PathRAG retrieval options
    p.add_argument("--retrieve", action="store_true", help="Enable PathRAG retrieval for user queries")
    p.add_argument("--k-seeds", type=int, default=4, help="Number of seed documents for PathRAG (default: 4)")
    p.add_argument("--beam", type=int, default=8, help="PathRAG beam width (default: 8)")
    p.add_argument("--hops", type=int, default=2, help="PathRAG max hops (default: 2)")
    p.add_argument("--fanout", type=int, default=3, help="PathRAG fanout per node (default: 3)")
    # GraphSAGE options
    p.add_argument("--graphsage", action="store_true", help="Enable GraphSAGE for retrieval (requires trained model)")
    p.add_argument("--graphsage-checkpoint", default="models/checkpoints/best.pt", help="Path to GraphSAGE checkpoint")
    p.add_argument("--graph-data", default="models/graph_data.pt", help="Path to graph export with node embeddings")
    # Core memory blocks
    p.add_argument("--persona", default="You are a helpful, concise engineer.")
    p.add_argument("--human", default="User: name=unknown. Preferences=unknown.")
    p.add_argument("--persona-file", default=None, help="Path to a persona template file (.md)")
    p.add_argument("--human-file", default=None, help="Path to a human profile template file (.md)")
    p.add_argument("--var", action="append", default=[], help="Template variable key=value (repeat)")
    # Runtime placement overrides
    p.add_argument("--device", default=None, help="Device string (e.g., cuda:0)")
    p.add_argument("--device-map", default=None, help="'balanced', 'auto', or explicit map (overrides env)")
    p.add_argument("--max-context", type=int, default=None, help="Max context tokens")
    p.add_argument("--max-memory", default=None, help="Comma list like '0=44GiB,1=44GiB' or 'cuda:0=44GiB,cuda:1=44GiB'")
    p.add_argument("--use-flash-attn", action="store_true", help="Request FlashAttention-2 if installed")
    return p.parse_args()


def build_orchestrator(args: argparse.Namespace) -> MemGPTOrchestrator:
    defaults = QwenModelConfig()
    model_id = os.getenv("QWEN_MODEL_ID", defaults.model_id)
    device = args.device or os.getenv("QWEN_DEVICE", defaults.device)
    device_map = args.device_map or (os.getenv("QWEN_DEVICE_MAP", "") or None)
    awq = os.getenv("QWEN_AWQ_WEIGHTS", defaults.awq_weights or "") or None
    max_ctx = args.max_context or int(os.getenv("QWEN_MAX_CONTEXT", str(defaults.max_context_length)))
    max_memory_env = args.max_memory or (os.getenv("QWEN_MAX_MEMORY", "") or None)
    use_flash = args.use_flash_attn or (os.getenv("QWEN_USE_FLASH_ATTN", "0").strip().lower() in {"1", "true", "yes", "on"})

    max_memory = None
    if max_memory_env:
        max_memory = {}
        for chunk in (c.strip() for c in max_memory_env.split(",") if c.strip()):
            k, v = chunk.split("=", 1)
            max_memory[k.strip()] = v.strip()

    config = QwenModelConfig(
        model_id=model_id,
        device=device,
        device_map=device_map,
        awq_weights=awq,
        max_memory=max_memory,
        max_context_length=max_ctx,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        use_flash_attn=use_flash,
    )
    telemetry = ConsoleTelemetryClient() if args.telemetry else None
    return MemGPTOrchestrator(config=OrchestratorConfig(auto_load_model=False), model_config=config, telemetry=telemetry)


def main() -> int:
    args = parse_args()

    orch = build_orchestrator(args)
    # Construct store; fail early with a helpful message if creds/socket are missing
    try:
        store = ArangoMemoryStore.from_env()
    except Exception as exc:
        print(f"[error] failed to construct Arango store: {exc}")
        return 1
    sess = MemGPTSession.new(orch, store, session_id=args.session or None)

    if args.dry_run:
        print("[plan] Ready to chat with:")
        print(f"  session_id={sess.session_id}")
        print(f"  model_id={orch.model_engine.config.model_id}")
        print(f"  device={orch.model_engine.config.device} device_map={orch.model_engine.config.device_map or 'default'}")
        if orch.model_engine.config.max_memory:
            mem = ",".join(f"{k}={v}" for k, v in orch.model_engine.config.max_memory.items())
            print(f"  max_memory={mem}")
        print(f"  max_new_tokens={orch.model_engine.config.max_new_tokens} temp={orch.model_engine.config.temperature} top_p={orch.model_engine.config.top_p}")
        if not args.no_schema:
            try:
                store.ensure_schema()
            except Exception as exc:
                print(f"[warn] ensure_schema failed: {exc}")
        print("[plan] No model load due to --dry-run")
        return 0

    if not args.no_schema:
        try:
            store.ensure_schema()
        except Exception as exc:
            print(f"[error] ensure_schema failed: {exc}")
            print("Hint: use --no-schema if collections already exist, or ensure ARANGO_* env vars are set (and server auth/socket are reachable).")
            return 1

    restored = 0
    try:
        restored = sess.restore_history(limit=args.restore)
    except Exception as exc:
        print(f"[warn] failed to restore history: {exc}")

    print(f"[chat] session={sess.session_id} restored_turns={restored}")

    engine = orch.model_engine
    try:
        engine.load()
    except Exception as exc:
        print(f"[error] Failed to load model: {exc}")
        return 1

    # Inject system prompt with core memory if missing
    # Resolve persona/human from files or flags with simple templating
    persona_text = args.persona
    human_text = args.human
    if args.persona_file or args.human_file or args.var:
        from core.runtime.memory.persona_loader import load_text, render_template  # lazy import
        vars_map = {}
        for pair in args.var:
            if "=" in pair:
                k, v = pair.split("=", 1)
                vars_map[k.strip()] = v.strip()
        if args.persona_file:
            persona_text = render_template(load_text(args.persona_file), vars_map)
        if args.human_file:
            human_text = render_template(load_text(args.human_file), vars_map)

    try:
        sess.ensure_system_core_memory(persona=persona_text, human=human_text, previous_message_count=restored)
    except Exception as exc:
        print(f"[warn] failed to ensure system memory: {exc}")

    # Initialize PathRAG if retrieval enabled
    pathrag = None
    embedder = None
    if args.retrieve:
        print("[chat] Initializing PathRAG retrieval...")
        try:
            cfg = resolve_memory_config()
            pathrag_client = ArangoMemoryClient(cfg)
            caps = PathRAGCaps(
                hops=args.hops,
                fanout=args.fanout,
                beam=args.beam,
            )

            # Initialize GraphSAGE if requested
            graphsage_inference = None
            if args.graphsage:
                if not GRAPHSAGE_AVAILABLE:
                    print("[warn] GraphSAGE requested but not available (missing dependencies)")
                else:
                    print("[chat] Loading GraphSAGE model...")
                    try:
                        import torch

                        # Load checkpoint and graph data
                        checkpoint_path = Path(args.graphsage_checkpoint)
                        graph_path = Path(args.graph_data)

                        if not checkpoint_path.exists():
                            print(f"[warn] GraphSAGE checkpoint not found: {checkpoint_path}")
                        elif not graph_path.exists():
                            print(f"[warn] Graph data not found: {graph_path}")
                        else:
                            # Load graph data for precomputed embeddings
                            graph_data = torch.load(graph_path, weights_only=False)

                            # Precompute node embeddings
                            graphsage_inference = GraphSAGEInference.from_checkpoint(
                                checkpoint_path=checkpoint_path,
                                device="cpu",
                            )

                            # Precompute embeddings for all nodes
                            node_embeddings = graphsage_inference.precompute_embeddings(
                                x=graph_data["data"].x,
                                edge_index_dict=graph_data["data"].edge_index_dict,
                            )

                            # Update inference with precomputed embeddings
                            node_ids = [graph_data["node_id_map"][i] for i in range(len(node_embeddings))]
                            graphsage_inference.node_embeddings = node_embeddings
                            graphsage_inference.node_ids = node_ids
                            graphsage_inference.node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

                            print(f"[chat] GraphSAGE ready with {len(node_ids)} precomputed embeddings")
                    except Exception as exc:
                        print(f"[warn] Failed to load GraphSAGE: {exc}")
                        import traceback
                        traceback.print_exc()

            pathrag = CodePathRAG(client=pathrag_client, caps=caps, graphsage_inference=graphsage_inference)
            embedder = JinaV4Embedder()

            if graphsage_inference:
                print(f"[chat] PathRAG ready with GraphSAGE: H={caps.hops}, F={caps.fanout}, B={caps.beam}")
            else:
                print(f"[chat] PathRAG ready: H={caps.hops}, F={caps.fanout}, B={caps.beam}")
        except Exception as exc:
            print(f"[warn] Failed to initialize PathRAG: {exc}")
            print("[warn] Continuing without retrieval")
            pathrag = None

    print("Type your message. Ctrl-D or /exit to quit.")
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue
        if user in {"/exit", ":q"}:
            break
        try:
            # PathRAG retrieval injection
            if pathrag and embedder:
                try:
                    # Generate query embedding
                    query_emb = embedder.embed_single(user, task="retrieval", prompt_name="query")

                    # Run PathRAG retrieval
                    result = pathrag.retrieve(
                        query=user,
                        query_embedding=query_emb.tolist(),
                        k_seeds=args.k_seeds
                    )

                    # Format retrieved context
                    if result.paths:
                        unique_docs = result.get_unique_documents()[:5]  # Top 5 docs
                        context_parts = []
                        for i, doc in enumerate(unique_docs, 1):
                            preview = doc.content[:600].replace('\n', ' ')
                            context_parts.append(f"[{i}] {doc.path}\n{preview}")

                        context_str = "\n\n".join(context_parts)
                        user_in = f"""Retrieved context from codebase (PathRAG, {result.latency_ms:.0f}ms):
{context_str}

User question: {user}"""
                        print(f"[retrieved] {len(unique_docs)} documents in {result.latency_ms:.0f}ms")
                    else:
                        user_in = user
                        print("[retrieved] No relevant documents found")
                except Exception as exc:
                    print(f"[warn] PathRAG retrieval failed: {exc}")
                    user_in = user
            else:
                user_in = user

            reply = sess.send_user_message(user_in)
        except Exception as exc:
            print(f"[error] generation failed: {exc}")
            continue
        print(f"bot> {reply.content}\n")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
