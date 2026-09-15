# 远程接入：建仓与同步分开确认

仅在第0步选择本地＋远程时读取。以下GitHub信息于2026-09-15按官方文档核对，实际使用前核对所选平台/接口的当前权限；不要求每位用户新建令牌。

## 优先复用现有条件

查看已有授权连接器、CLI登录状态、Git remote配置与认证可用性。浏览器只查看正常页面登录状态，不读取cookie或会话密钥。凭据管理器/SSH agent可以让AI调用Git而无需看到秘密。读取公开仓库成功只证明读取可用；真正推送以批准后的首次push及远端SHA核验为准。

| 方式 | 创建远程仓库 | 本地同步 |
|---|---|---|
| 已登录浏览器 | 用户授权AI操作建仓表单 | 另需本机SSH或HTTPS认证 |
| 用户提供空仓库URL | 用户自行完成 | 用户配置本机密钥/凭据并授权使用 |
| 已授权CLI/连接器或PAT | 检查其建仓能力及归属 | 检查Git传输认证及目标仓库写权限 |

## GitHub准确名称和权限

**Personal Access Token（PAT，个人访问令牌）**可用于API与HTTPS Git认证；GitHub建议可行时优先使用细粒度PAT。它不用于SSH URL。可通过GitHub CLI或凭据管理器保存认证，避免把令牌嵌入命令或remote URL。[官方PAT说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

个人仓库API建仓接口 `POST /user/repos` 支持细粒度PAT，要求仓库权限 **Administration: write**。classic PAT创建公开仓库需要 `public_repo` 或 `repo`，创建私有仓库需要 `repo`。具体接口和账号能力共同限制操作，令牌不能授予账号本身没有的能力。[官方建仓接口](https://docs.github.com/en/rest/repos/repos#create-a-repository-for-the-authenticated-user)

本地HTTPS推送还需要对目标仓库的内容写权限；细粒度PAT按需配置 **Contents: write**。检查resource owner、repository access是否覆盖新仓库，以及组织批准/SSO和分支规则。不要因为有Administration权限就假定可以推送；也不要为了建仓默认请求全部仓库长期写权限。若无法适当覆盖尚未创建的仓库，可选浏览器/用户建空仓库，再针对它配置推送权限。工作流文件等额外操作按实际需要核对权限。[官方细粒度权限表](https://docs.github.com/en/rest/authentication/permissions-required-for-fine-grained-personal-access-tokens)

**SSH key（SSH密钥）**由公钥与私钥组成，不是“SSH令牌”。平台登记公钥，本机私钥/SSH agent完成Git认证；SSH可读写已授权仓库，但不能单凭SSH密钥调用建仓REST接口。[官方SSH说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh)

## 交付关联与同步

审批单确认平台、归属、可见性与同步规则后再创建/连接。已有remote先核对，不盲目覆盖；设置对应upstream。首次push后记录本地提交与远端SHA一致。后续按约定fetch、整合远端变化、提交和推送；失败时保留本地工作并报告未同步状态。定时或自动同步需要单独明确安排，不能宣称设置remote即实现自动双向同步。
