"""Real SDK verification with an ephemeral test CA, never installed in production."""
import io
import struct
import zlib
from datetime import datetime,timedelta,timezone
import c2pa
from cryptography import x509
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
from PIL import Image
from app import provenance


def test_signed_binding_and_tampering(tmp_path,monkeypatch):
    root_key=ec.generate_private_key(ec.SECP256R1());leaf_key=ec.generate_private_key(ec.SECP256R1())
    root_name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Portrait TEST ONLY Root')])
    leaf_name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Portrait TEST ONLY Signer')])
    now=datetime.now(timezone.utc)
    def builder(subject,issuer,key,ca,authority):
        return (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(days=1)).not_valid_after(now+timedelta(days=7))
            .add_extension(x509.BasicConstraints(ca=ca,path_length=None),critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True,content_commitment=not ca,key_encipherment=False,data_encipherment=False,key_agreement=False,key_cert_sign=ca,crl_sign=ca,encipher_only=False,decipher_only=False),critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(authority.public_key()),critical=False))
    root=builder(root_name,root_name,root_key,True,root_key).sign(root_key,hashes.SHA256())
    leaf=(builder(leaf_name,root_name,leaf_key,False,root_key)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]),critical=True).sign(root_key,hashes.SHA256()))
    root_pem=root.public_bytes(serialization.Encoding.PEM)
    certs=leaf.public_bytes(serialization.Encoding.PEM)+root_pem
    key=leaf_key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption())
    models=tmp_path/'models';models.mkdir();(models/'c2pa-trust-anchors.pem').write_bytes(root_pem)
    monkeypatch.setattr(provenance,'ROOT',tmp_path)
    settings=c2pa.Settings.from_dict({'trust':{'trust_anchors':root_pem.decode()},'verify':{'remote_manifest_fetch':False,'ocsp_fetch':False},'core':{'allowed_network_hosts':[]}})
    source=io.BytesIO();Image.new('RGB',(40,40),'blue').save(source,'PNG');source.seek(0);dest=io.BytesIO()
    manifest={'claim_generator_info':[{'name':'Portrait test fixture'}],'title':'test.png','format':'image/png','assertions':[{'label':'c2pa.actions.v2','data':{'actions':[{'action':'c2pa.created','digitalSourceType':'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'}]}}]}
    with c2pa.Signer.from_info(c2pa.C2paSignerInfo(b'es256',certs,key,None)) as signer:
        with c2pa.Context(settings=settings) as context:
            with c2pa.Builder(manifest,context=context) as builder:
                builder.sign(signer,'image/png',source,dest)
    path=tmp_path/'signed.png';path.write_bytes(dest.getvalue())
    result=provenance.provenance_report(path)
    assert result['signature_status']=='valid'
    assert result['asset_binding_status']=='valid'
    assert result['signer_trust']=='trusted',result
    assert result['category']=='ai_generated',result
    data=path.read_bytes();output=data[:8];offset=8
    while offset<len(data):
        length=struct.unpack('>I',data[offset:offset+4])[0];kind=data[offset+4:offset+8];chunk=data[offset+8:offset+8+length]
        if kind==b'IDAT':
            raw=bytearray(zlib.decompress(chunk));raw[1]^=1;chunk=zlib.compress(bytes(raw))
        output+=struct.pack('>I',len(chunk))+kind+chunk+struct.pack('>I',zlib.crc32(kind+chunk)&0xffffffff)
        offset+=12+length
    path.write_bytes(output)
    tampered=provenance.provenance_report(path)
    assert tampered['asset_binding_status']=='invalid',tampered
    assert tampered['category']=='unknown' and tampered['platform'] is None
