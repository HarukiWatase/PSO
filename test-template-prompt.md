# テストファイル作成プロンプトテンプレート

@実装対象ファイル (例: src/infrastructure/ExampleRepository.ts)
@基準ファイル (src/infrastructure/billingMessageTemplateRepository.test.ts)

上記 `[実装対象ファイル]` のテストコードを新規作成してください。
実装にあたっては、`[基準ファイル]` の実装スタイルを「絶対的な正解（Gold Standard）」とし、以下の指示と規約を厳守してください。

## 🕵️‍♂️ 実装検証（QA視点・最重要）
**テストを書くだけでなく、対象コードの実装自体が正しいかも検証してください。**
- もし実装コードにバグ（引数ミス、Prismaメソッドの誤用、不適切な型変換など）が見つかった場合は、テストを通すために合わせるのではなく、**実装コード側を修正してください。**
- 「テストが通る＝正しい」ではなく、「仕様として正しいか」を常に意識してください。

## 🛡️ 全12項目の鉄の掟（実装規約）

### 1. インフラストラクチャ・設定 (安定性)
1. **環境指定ヘッダー**: ファイルの **1行目** に `// @vitest-environment node` を記述すること。
2. **ReferenceError完全回避とモックの適切な管理**: 
   - **Infrastructure層のテスト（Prisma使用）の場合**:
     - `vi.hoisted` + `mockDeep` パターンを使用すること。テスト内で直接操作する必要があるモック関数（例: `mockClerkClientFn`, `mockPrismaFn` など）は、`vi.hoisted` で外に出すこと。これにより、テストケースごとに `mockResolvedValue` や `mockRejectedValue` を書き換える際に直接操作でき、より直感的で保守しやすいコードになる。
     - **`vi.hoisted` を使う理由**: `vi.hoisted` を使うことで、`vi.mock` よりも先に初期化されるため、テストケース内で直接モックを操作できるようになる。動的インポートを使わずに済むため、コードがシンプルになる。
     - `vi.mock` ファクトリ内では: 外部変数が参照できないため、必ず `await import` を行う「動的インポート方式」を採用すること。`vi.hoisted` で定義したモック関数に、`mockDeep` で作成したモックオブジェクトを設定する。
     - テストケース（`it`）内では: 動的インポートを使わず、`vi.hoisted` で定義したモック関数を直接操作すること。ただし、トランザクション用のモックを作成する場合は、`it` ブロック内で動的インポートを使用する（`vi.mock` ファクトリ内では外部変数が参照できないため）。
     - **`@prisma/client` のモック処理**: `PrismaClient` コンストラクタをモックし、`Prisma` 名前空間（型・ユーティリティ）は元のまま使用すること。`vi.importActual` を使用して `Prisma.Decimal` 等を保持すること。
   - **UseCase層・Presentation層のテストの場合**:
     - **DIパターン（直接注入）を使用すること**。`vi.mock` や `vi.hoisted` は使用しない。
     - `it` ブロック内で `mockDeep<InterfaceType>()` を使ってモックを作成し、Factory関数の引数として直接注入すること。
     - 複数の依存関係がある場合は、`setup()` 関数パターンを使用すること。
3. **ノイズ抑制**: `beforeEach` 内に `vi.spyOn(console, "error").mockImplementation(() => {})` を記述し、テスト実行時の意図的なエラーログを抑制すること。
4. **テストの独立性とモックのリセット**: 
   - Infrastructure層のテストでは: `beforeEach` 内で `mockReset(prismaMock)` を使用すること（`vi.clearAllMocks()` ではなく）。`mockReset` は履歴だけでなく実装（`mockResolvedValue` 等）もリセットするため、テスト間の汚染を防げる。
   - UseCase層のテストでは: `it` ブロック内でモックを作成するため、基本的に `beforeEach` でのリセットは不要。ただし、`beforeEach` でセットアップ関数を使う場合は、必要に応じて `mockReset()` を使用すること。

### 2. コード品質・設計パターン (保守性)
5. **Factoryパターンの強制**: テストデータ生成は `createTestEntity` 等のヘルパー関数に切り出し、`it` ブロック内でのオブジェクト直書きは避けること。将来的に `.factory.ts` ファイルに切り出す可能性を考慮し、データ生成ロジックのみを切り出すこと（モック実装は含めない）。
6. **日時の決定論的記述**: 
   - `new Date()` を直接使わず、定数 `TEST_DATE` 等を使用して日時を固定化すること（Flaky test防止）。
   - 時間に依存するテストでは `vi.useFakeTimers()` と `vi.setSystemTime()` を使用すること。
   - `afterEach` で必ず `vi.useRealTimers()` を呼び出してタイマーを元に戻すこと。
7. **型のシンプル化**: 複雑な `GetPayload` 型ではなく、`Prisma.ModelName` (モデル型) を使用すること。
8. **モックの実装方法（スキーマ変更対応優先）**: 
   - **原則**: スキーマ変更に対応できる実装を優先すること。これは、Prisma スキーマの変更や外部サービスの API 変更時にテストコードの修正を最小限に抑えるため。
   - **Infrastructure層の場合**: `mockDeep<PrismaClient>()` を使用すること。手動で型定義を書く（例: `type MockBillingModel = { ... }`）ことは避けること。インターフェースから自動的にモックを生成することで、型安全性と保守性を確保できる。
   - **UseCase層の場合**: `mockDeep<RepositoryInterface>()` を使用すること。Domain層で定義されたインターフェースから自動的にモックを生成する。
   - **型アサーション**: `as unknown as DeepMockProxy<T>` はモック初期化時の1回のみとし、テスト内での Non-null assertion (`!`) の多用は避けること。
   - **`.factory.ts` の役割**: テストデータ生成のみを担当すること。モック実装（`vi.fn()` 等）は含めないこと。これにより、データ生成とモック実装の役割が明確に分離され、保守性が向上する。

### 3. テストロジック・網羅性 (品質保証)
9. **Resultパターンの検証**: 戻り値検証では `expect(result.ok).toBe(...)` と `result.value/error` の中身まで検証すること。`if (result.ok)` または `if (!result.ok)` による型ガードを必ず使用すること。
10. **引数の厳密な検証**: `toHaveBeenCalledWith` を使い、Prismaに渡された検索条件や更新データの中身まで厳密に検証すること。ただし、クエリ最適化で壊れる「脆いテスト」にならないよう、呼び出し検証は「副作用（保存・削除）がある場合」のみにすること。
11. **トランザクションの厳格な分離**: 
   - トランザクションテストにおいて、`mockPrisma`(通常) と `mockTx`(Tx用) の呼び出しが混同されていないか検証すること。
   - トランザクション内でリポジトリを使用する場合は、`it` ブロック内で `mockDeep<TransactionClient>()` を使って新しいモックを作成し、RepositoryのFactory関数に直接渡すこと。
   - **動的インポートの使用**: `vi.mock` ファクトリ内では外部変数が参照できないため、トランザクション用のモックを作成する際は `it` ブロック内で `await import("vitest-mock-extended")` を使って動的インポートを行うこと。
   - 通常の `prisma` とトランザクションクライアント `tx` の呼び出しが混同されていないか、`expect(mockPrisma.model.method).not.toHaveBeenCalled()` で検証すること。
12. **網羅率100%**: 正常系だけでなく、DBエラー・存在しないID・バリデーションエラーなどの異常系を網羅し、**ブランチカバレッジ 100%** を達成すること。Prisma固有エラー（`P2002`, `P2025` 等）のハンドリングもテストすること。

## 📋 レイヤー別実装パターン

### Infrastructure層のテスト（Prisma使用）の場合

```typescript
// @vitest-environment node
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mockDeep, mockReset, DeepMockProxy } from "vitest-mock-extended";
import { PrismaClient, Prisma } from "@prisma/client";

// =============================================================================
// モジュールモック（インポートの前に配置）
// =============================================================================

// 1. @prisma/client のコンストラクタをモック
// PrismaClientだけをモックし、Prisma名前空間（型・ユーティリティ）は元のまま使用
vi.mock("@prisma/client", async () => {
	const actual =
		await vi.importActual<typeof import("@prisma/client")>("@prisma/client");
	return {
		...actual,
		PrismaClient: vi.fn(() => ({
			$connect: vi.fn(),
			$disconnect: vi.fn(),
			$transaction: vi.fn(),
		})),
	};
});
vi.mock("@prisma/adapter-pg", () => ({}));
vi.mock("dotenv/config", () => ({}));

// 2. vi.hoisted でモック定義（ホイスティング保証）
// 【重要】vi.hoisted を使うことで、vi.mock よりも先に初期化されるため、
// テストケース内で直接 prismaMock を操作できるようになる。
// これにより、テストケースごとに mockResolvedValue や mockRejectedValue を
// 書き換える際に、動的インポートを使わずに直接操作でき、より直感的で保守しやすいコードになる。
// 
// 注意: 基準ファイル（billingMessageTemplateRepository.test.ts）では
// vi.hoisted を使わず vi.mock ファクトリ内で直接 mockDeep() を呼んでいますが、
// 今後は vi.hoisted パターンで統一するため、このテンプレートでは vi.hoisted を推奨しています。
const { prismaMock } = vi.hoisted(() => ({
	prismaMock: mockDeep<PrismaClient>(),
}));

// 3. @/src/lib/prisma のインスタンスを mockDeep でモック
// vi.importActual で Prisma 名前空間を保持（Prisma.Decimal 等に必須）
vi.mock("@/src/lib/prisma", async () => {
	const actual =
		await vi.importActual<typeof import("@prisma/client")>("@prisma/client");
	return {
		...actual,
		prisma: prismaMock,
	};
});

// =============================================================================
// インポート（モック適用後）
// =============================================================================
import { newYourRepository } from "./yourRepository";
import type { TransactionClient } from "@/src/lib/prisma";

// テスト内で扱いやすいように型アサーション（実体は mockDeep されたもの）
const mockPrisma = prismaMock as unknown as DeepMockProxy<PrismaClient>;

// =============================================================================
// テストヘルパー / Factory関数
// =============================================================================

/** テスト用の固定日時 */
const TEST_DATE = new Date("2024-01-01T00:00:00Z");

/** テスト用のRecordを作成（DBから返される形式） */
function createTestRecord(overrides?: {
	id?: string;
	// ... その他のフィールド
}): YourModel {
	return {
		id: overrides?.id ?? "test-id",
		// ... デフォルト値
		createdAt: TEST_DATE,
		updatedAt: TEST_DATE,
	};
}

/**
 * Prisma.Decimal を使用したテストデータ生成の例
 * 金額などの計算に使用する場合は、必ず Prisma.Decimal を使用すること
 */
function createTestBillingRecord(
	overrides?: Partial<{
		id: string;
		amount: number | Prisma.Decimal;
	}>
): {
	id: string;
	amount: Prisma.Decimal;
	createdAt: Date;
	updatedAt: Date;
} {
	return {
		id: overrides?.id ?? "test-id",
		amount:
			overrides?.amount !== undefined
				? typeof overrides.amount === "number"
					? new Prisma.Decimal(overrides.amount)
					: overrides.amount
				: new Prisma.Decimal(1000),
		createdAt: TEST_DATE,
		updatedAt: TEST_DATE,
	};
}

/**
 * Prisma 固有エラーを正しく生成するヘルパー
 * Bun ランタイムでの `instanceof` 問題を回避するため、プロトタイプを明示的に設定
 */
function createPrismaError(
	code: string,
	message: string = "Database error",
	meta?: Record<string, unknown>
): Prisma.PrismaClientKnownRequestError {
	const error = new Prisma.PrismaClientKnownRequestError(message, {
		code,
		clientVersion: "7.0.0",
		meta,
	});
	// Bun 環境での instanceof チェックを通過させるため
	Object.setPrototypeOf(error, Prisma.PrismaClientKnownRequestError.prototype);
	return error;
}

// =============================================================================
// テスト本体
// =============================================================================
describe("YourRepository", () => {
	beforeEach(() => {
		// 決定論的実行: システムクロックを固定
		vi.useFakeTimers();
		vi.setSystemTime(TEST_DATE);

		// mockReset で履歴だけでなく実装（戻り値）もリセット
		mockReset(prismaMock);
		// テスト中の意図的なエラーログ（console.error）を出力しないように抑制
		vi.spyOn(console, "error").mockImplementation(() => {});
	});

	afterEach(() => {
		// タイマーを元に戻す
		vi.useRealTimers();
	});

	describe("getById", () => {
		it("IDで正しくデータを取得できること", async () => {
			const record = createTestRecord({ id: "123" });
			mockPrisma.yourModel.findUnique.mockResolvedValue(record as any);

			const repository = newYourRepository();
			const result = await repository.getById("123");

			expect(result.ok).toBe(true);
			if (result.ok && result.value) {
				expect(result.value.id).toBe("123");
			}
		});

		it("存在しない場合はnullを返すこと", async () => {
			mockPrisma.yourModel.findUnique.mockResolvedValue(null);

			const repository = newYourRepository();
			const result = await repository.getById("not-found");

			expect(result.ok).toBe(true);
			if (result.ok) {
				expect(result.value).toBeNull();
			}
		});

		it("DBエラー時はエラーを返すこと", async () => {
			// Prisma 固有エラーを正しく生成（Bun 環境での instanceof 問題を回避）
			const prismaError = createPrismaError(
				"P2002",
				"Unique constraint failed",
				{
					target: ["id"],
				}
			);
			mockPrisma.yourModel.findUnique.mockRejectedValue(prismaError);

			const repository = newYourRepository();
			const result = await repository.getById("123");

			expect(result.ok).toBe(false);
			if (!result.ok) {
				expect(result.error.kind).toBe("internal");
			}
		});
	});

	describe("トランザクション", () => {
		it("トランザクションが使用された場合、正しく処理が行われること", async () => {
			const record = createTestRecord({ id: "123" });

			// 【重要】mockDeepを動的インポートして作成
			// vi.mock ファクトリ内では外部変数が参照できないため、
			// it ブロック内で動的インポート（await import）を使用する必要がある。
			// これにより、トランザクション用の新しいモックインスタンスを作成できる。
			const { mockDeep } = await import("vitest-mock-extended");
			const mockTx = mockDeep<TransactionClient>();
			mockTx.yourModel.create.mockResolvedValue(record as any);

			// トランザクション内でリポジトリを使用
			const repository = newYourRepository(mockTx);
			const result = await repository.save({
				/* entity */
			});

			expect(result.ok).toBe(true);
			// トランザクションクライアントのcreateが呼ばれたことを確認
			expect(mockTx.yourModel.create).toHaveBeenCalled();
			// 通常のprismaのcreateは呼ばれていないことを確認
			expect(mockPrisma.yourModel.create).not.toHaveBeenCalled();
		});

		it("トランザクション内でエラーが発生した場合、エラーが伝播されること（ロールバック検証）", async () => {
			// トランザクション用の新しいモックインスタンスを作成
			// （動的インポートの理由は上記のテストケースを参照）
			const { mockDeep } = await import("vitest-mock-extended");
			const mockTx = mockDeep<TransactionClient>();

			const dbError = new Error("Database connection error");
			mockTx.yourModel.create.mockRejectedValue(dbError);

			const repository = newYourRepository(mockTx);
			const result = await repository.save({
				/* entity */
			});

			expect(result.ok).toBe(false);
			if (!result.ok) {
				expect(result.error.kind).toBe("internal");
			}
			expect(mockTx.yourModel.create).toHaveBeenCalled();
			expect(mockPrisma.yourModel.create).not.toHaveBeenCalled();
		});
	});
});
```

### UseCase層のテストの場合

```typescript
// @vitest-environment node
import { describe, it, expect } from "vitest";
import { mockDeep } from "vitest-mock-extended";
import { newYourUseCase } from "./yourUseCase";
import { RepositoryInterface } from "@/src/domain/your/repository";
import { ok, err } from "@/src/lib/result";

describe("YourUseCase", () => {
	// テストごとに独立したメモリアドレスを割り振る「工場」
	const setup = () => {
		const mockRepository = mockDeep<RepositoryInterface>();
		const usecase = newYourUseCase(mockRepository);
		return { usecase, mockRepository };
	};

	it("正常系：正しく処理が行われること", async () => {
		const { usecase, mockRepository } = setup();

		// モックの挙動を設定
		mockRepository.getById.mockResolvedValue(ok(someData));

		// 実行
		const result = await usecase.execute({ id: "123" });

		// 検証
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.value).toBeDefined();
		}
		expect(mockRepository.getById).toHaveBeenCalled();
	});

	it("異常系：エラーが返されること", async () => {
		const { usecase, mockRepository } = setup();

		mockRepository.getById.mockResolvedValue(err(newInternalErr()));

		const result = await usecase.execute({ id: "123" });

		expect(result.ok).toBe(false);
		if (!result.ok) {
			expect(result.error.kind).toBe("internal");
		}
	});
});
```

## ⚠️ 重要なアンチパターン（避けるべき）

- ❌ `vi.clearAllMocks()` の使用 → ✅ `mockReset()` を使用すること
- ❌ Infrastructure層で手動型定義（`type MockBillingModel = { ... }`）→ ✅ `mockDeep<PrismaClient>()` を使用すること
- ❌ UseCase層で `vi.mock` を使用 → ✅ DIパターン（直接注入）を使用すること
- ❌ `.factory.ts` にモック実装（`vi.fn()`）を含める → ✅ データ生成のみを配置すること
- ❌ Prisma固有エラーを標準 `Error` でモック → ✅ `Prisma.PrismaClientKnownRequestError` を使用すること（Bun環境では `Object.setPrototypeOf` も必要）
- ❌ `vi.importActual` を省略 → ✅ `Prisma.Decimal` 等を保持するため必須
- ❌ 日時固定のテストで `afterEach` で `vi.useRealTimers()` を呼ばない → ✅ 必ず呼び出すこと
- ❌ トランザクションテストで通常の `prisma` と `tx` の呼び出しを混同 → ✅ `expect(mockPrisma.model.method).not.toHaveBeenCalled()` で検証すること

## 📚 参考ファイル

実装時に以下のファイルを参照してください：

- **基準ファイル**: `src/infrastructure/billingMessageTemplateRepository.test.ts`
  - 注意: このファイルでは `vi.hoisted` を使わず `vi.mock` ファクトリ内で直接 `mockDeep()` を呼んでいますが、今後は `vi.hoisted` パターンで統一するため、このテンプレートの推奨パターンを優先してください。
- **Infrastructure層のモックパターン**: `src/infrastructure/clerkMetadataService.test.ts`（`vi.hoisted` でモック関数を定義するパターン）
  - このファイルは `vi.hoisted` パターンの参考として最適です。
- **UseCase層のDIパターン**: `src/usecase/billing/updateBilling.test.ts`
- **テストテンプレート詳細**: `.claude/commands/test-template.md`
- **テストガイド**: `docs/test/vitest-testing-guide.md`

## 出力

作成したテストコード全体を出力してください。
